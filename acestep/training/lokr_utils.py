"""
LoKr Utilities for ACE-Step

Provides utilities for injecting LoKr (Low-Rank Kronecker) adapters into the
DiT decoder model via the LyCORIS library. LoKr uses Kronecker product
factorization as an alternative to standard LoRA low-rank decomposition.

Requires: pip install lycoris-lora
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

import torch
import torch.nn as nn

try:
    from lycoris import create_lycoris, create_lycoris_from_weights, LycorisNetwork
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    logger.info("LyCORIS library not installed. LoKr training will not be available. "
                "Install with: pip install lycoris-lora")


def check_lycoris_available() -> bool:
    """Check if LyCORIS library is available."""
    return LYCORIS_AVAILABLE


def _build_target_regex(target_modules: List[str], attention_type: str = "both", train_mlp: bool = False) -> str:
    """Build a regex pattern for LyCORIS apply_preset from target module names.

    LyCORIS uses regex-based target matching via apply_preset(), unlike PEFT
    which accepts a list of module short names.

    Args:
        target_modules: List of module short names (e.g. ["q_proj", "k_proj", "v_proj", "o_proj"])
        attention_type: "both", "self", or "cross"
        train_mlp: Whether to also match MLP layers
    
    Returns:
        Regex pattern string for LyCORIS target_name preset
    """
    # Build module name alternatives: q_proj|k_proj|v_proj|o_proj
    modules_pattern = "|".join(target_modules)
    
    # MLP pattern
    mlp_pattern = ""
    if train_mlp:
        mlp_pattern = "|.*mlp\\.(gate_proj|up_proj|down_proj)"

    if attention_type == "self":
        # Only self-attention layers
        pattern = f".*self_attn.*({modules_pattern})$"
    elif attention_type == "cross":
        # Only cross-attention layers
        pattern = f".*cross_attn.*({modules_pattern})$"
    else:
        # Both — match any attention layer with these projections
        pattern = f".*({modules_pattern})$"
    
    if train_mlp:
        pattern = f"({pattern}){mlp_pattern}"
    
    return pattern


def inject_lokr_into_dit(
    model,
    lokr_config,
    attention_type: str = "both",
    train_mlp: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Inject LoKr adapters into the DiT decoder of the model.

    Uses the LyCORIS library to create LoKr (Kronecker product) adapters.
    Unlike PEFT LoRA which replaces model.decoder with a PeftModel wrapper,
    LyCORIS injects hooks into the existing model and returns a separate
    LycorisNetwork object that manages the adapter parameters.

    Args:
        model: The AceStepConditionGenerationModel
        lokr_config: LoKRConfig with linear_dim, linear_alpha, factor, etc.
        attention_type: "both", "self", or "cross" — which attention layers to target

    Returns:
        Tuple of (model, lycoris_network, info_dict)
        - model: The original model (decoder is NOT replaced, LyCORIS hooks into it)
        - lycoris_network: LycorisNetwork object managing the LoKr adapter
        - info_dict: Parameter counts and config info
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoKr training. "
            "Install with: pip install lycoris-lora"
        )

    # Check for quantization (reuse from lora_utils)
    from acestep.training.lora_utils import check_model_quantized
    quant_type = check_model_quantized(model)
    if quant_type != "none":
        if quant_type in ("bitsandbytes", "int_weights"):
            raise RuntimeError(
                f"Cannot inject LoKr into a {quant_type}-quantized model. "
                "Integer-quantized weights cannot be upcasted for training. "
                "Please restart the service without quantization."
            )
        logger.warning(
            f"Model has {quant_type} quantization. "
            "Training quality may be slightly reduced vs. a full-precision checkpoint."
        )

    # Build target module regex for LyCORIS
    target_modules = lokr_config.target_modules
    
    # Sync target_modules metadata if MLP training is requested
    if train_mlp:
        mlp_projs = ["gate_proj", "up_proj", "down_proj"]
        for proj in mlp_projs:
            if proj not in target_modules:
                target_modules.append(proj)
        logger.info(f"Expanded LoKr target_modules metadata to include MLP: {target_modules}")

    target_regex = _build_target_regex(target_modules, attention_type, train_mlp=train_mlp)

    logger.info(f"LoKr target pattern: {target_regex}")
    logger.info(f"LoKr config: dim={lokr_config.linear_dim}, alpha={lokr_config.linear_alpha}, "
                f"factor={lokr_config.factor}, decompose_both={lokr_config.decompose_both}, "
                f"use_tucker={lokr_config.use_tucker}")

    # Configure LyCORIS preset for targeting specific modules
    LycorisNetwork.apply_preset({
        "target_name": [target_regex],
    })

    # Create LoKr network
    decoder = model.decoder

    lokr_kwargs = {
        "algo": "lokr",
        "factor": lokr_config.factor,
        "decompose_both": lokr_config.decompose_both,
        "use_tucker": lokr_config.use_tucker,
    }

    # Only pass dropout if > 0
    if lokr_config.dropout > 0:
        lokr_kwargs["dropout"] = lokr_config.dropout

    lycoris_net = create_lycoris(
        decoder,
        multiplier=1.0,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        **lokr_kwargs,
    )

    # Apply the LoKr hooks to the decoder
    lycoris_net.apply_to()

    # CRITICAL: LyCORIS creates adapter parameters on CPU regardless of where
    # the decoder lives. We must explicitly move everything to the decoder's
    # device so all parameters (base + adapter) are co-located. Without this,
    # every forward pass triggers CPU→GPU transfers, making training ~10-50x slower.
    decoder_device = next(decoder.parameters()).device
    lycoris_net = lycoris_net.to(decoder_device)
    # Also ensure any new params injected into the decoder are on the right device
    model.decoder = model.decoder.to(decoder_device)
    logger.info(f"LoKr adapter parameters moved to {decoder_device}")

    # Freeze all base model parameters, keep only LoKr params trainable
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradients for LyCORIS parameters
    for param in lycoris_net.parameters():
        param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in lycoris_net.parameters())
    lokr_module_count = len(lycoris_net.loras) if hasattr(lycoris_net, 'loras') else 0

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "lokr_modules": lokr_module_count,
        "linear_dim": lokr_config.linear_dim,
        "linear_alpha": lokr_config.linear_alpha,
        "factor": lokr_config.factor,
        "target_pattern": target_regex,
    }

    logger.info(f"LoKr injected into DiT decoder:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({info['trainable_ratio']:.2%})")
    logger.info(f"  LoKr modules: {lokr_module_count}")
    logger.info(f"  Factor: {lokr_config.factor}, Decompose both: {lokr_config.decompose_both}")

    return model, lycoris_net, info


def save_lokr_weights(
    lycoris_net,
    output_dir: str,
    lokr_config=None,
    dtype=None,
    trigger_word: str = "",
    tag_position: str = "",
) -> str:
    """Save LoKr adapter weights.

    Saves the LyCORIS network weights as safetensors and a lokr_config.json
    metadata file for later reloading and adapter type detection.

    Args:
        lycoris_net: The LycorisNetwork object
        output_dir: Directory to save weights
        lokr_config: Optional LoKRConfig for saving metadata
        dtype: Optional dtype for saving (e.g. torch.bfloat16)
        trigger_word: Activation tag / trigger word for this LoRA style
        tag_position: How the trigger word was used during training

    Returns:
        Path to saved adapter directory
    """
    os.makedirs(output_dir, exist_ok=True)

    adapter_path = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_path, exist_ok=True)

    # Save weights using LyCORIS native format
    weights_file = os.path.join(adapter_path, "lokr_weights.safetensors")
    lycoris_net.save_weights(
        weights_file,
        dtype=dtype or torch.bfloat16,
        metadata={"adapter_type": "lokr"},
    )

    # Save config metadata for reload/detection
    config_data = {
        "adapter_type": "lokr",
        "algo": "lokr",
    }
    if lokr_config is not None:
        config_data.update(lokr_config.to_dict())
    if trigger_word:
        config_data["trigger_word"] = trigger_word
        if tag_position:
            config_data["tag_position"] = tag_position

    config_path = os.path.join(adapter_path, "lokr_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    logger.info(f"LoKr adapter saved to {adapter_path}")
    return adapter_path


def load_lokr_weights(
    model,
    lokr_path: str,
    lokr_config=None,
) -> Tuple[Any, Any]:
    """Load LoKr adapter weights into the model.

    Args:
        model: The base model (without LoKr)
        lokr_path: Path to saved LoKr weights directory
        lokr_config: Optional LoKRConfig

    Returns:
        Tuple of (model, lycoris_network)
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError("LyCORIS is required to load LoKr weights. Install: pip install lycoris-lora")

    if not os.path.exists(lokr_path):
        raise FileNotFoundError(f"LoKr weights not found: {lokr_path}")

    # Find the weights file
    if os.path.isdir(lokr_path):
        weights_file = os.path.join(lokr_path, "lokr_weights.safetensors")
        if not os.path.exists(weights_file):
            # Check in adapter subfolder
            weights_file = os.path.join(lokr_path, "adapter", "lokr_weights.safetensors")
    else:
        weights_file = lokr_path

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"LoKr weights file not found: {weights_file}")

    # Load weights into a LyCORIS network
    lycoris_net, info = create_lycoris_from_weights(
        multiplier=1.0,
        file=weights_file,
        module=model.decoder,
    )
    lycoris_net.apply_to()
    lycoris_net.load_weights(weights_file)

    logger.info(f"LoKr weights loaded from {weights_file}")
    return model, lycoris_net


def merge_lokr_weights(model, lycoris_net, weight: float = 1.0) -> Any:
    """Merge LoKr weights into the base model permanently.

    After merging, the model can be used without LyCORIS.

    Args:
        model: Model with LoKr adapters applied
        lycoris_net: The LycorisNetwork managing the adapters
        weight: Merge weight (1.0 = full merge)

    Returns:
        Model with merged weights
    """
    lycoris_net.merge_to(weight)
    logger.info("LoKr weights merged into base model")
    return model


def save_lokr_training_checkpoint(
    lycoris_net,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    lokr_config=None,
    trigger_word: str = "",
    tag_position: str = "",
) -> str:
    """Save a training checkpoint including LoKr weights and training state.

    Args:
        lycoris_net: The LycorisNetwork
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        global_step: Current global step
        output_dir: Directory to save checkpoint
        lokr_config: Optional LoKRConfig for metadata
        trigger_word: Activation tag / trigger word for this LoRA style
        tag_position: How the trigger word was used during training

    Returns:
        Path to saved checkpoint directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save LoKr adapter weights
    adapter_path = save_lokr_weights(lycoris_net, output_dir, lokr_config=lokr_config, trigger_word=trigger_word, tag_position=tag_position)

    # Save training state (optimizer, scheduler, epoch, step)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "adapter_type": "lokr",
    }

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(training_state, state_path)

    logger.info(f"LoKr checkpoint saved to {output_dir} (epoch {epoch}, step {global_step})")
    return output_dir
