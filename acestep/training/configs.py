"""
Training Configuration Classes

Contains dataclasses for LoRA and training configurations.
Includes VRAM-aware presets loaded from external JSON files.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
from loguru import logger


# Directory containing preset JSON files
_PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")


def detect_best_precision() -> str:
    """Auto-detect the best training precision for the current device.

    Returns:
        Precision string: "bf16", "fp16", or "fp32"
    """
    if torch.cuda.is_available():
        # Check if GPU supports bfloat16 (Ampere+ / SM 80+)
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            return "bf16"
        else:
            # Pre-Ampere GPUs (e.g. GTX 1080, RTX 2080) - use fp16
            logger.info(f"GPU compute capability {capability[0]}.{capability[1]} < 8.0, using fp16 instead of bf16")
            return "fp16"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        # Intel XPU - supports bf16
        return "bf16"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon MPS - fp16 is more reliable than bf16
        return "fp16"
    else:
        # CPU fallback
        return "fp32"


def precision_to_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype.

    Args:
        precision: One of "bf16", "fp16", "fp32"

    Returns:
        Corresponding torch.dtype
    """
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(precision, torch.float32)


def precision_to_fabric_str(precision: str) -> str:
    """Convert precision string to Lightning Fabric precision string.

    Args:
        precision: One of "bf16", "fp16", "fp32"

    Returns:
        Fabric-compatible precision string
    """
    mapping = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true",
    }
    return mapping.get(precision, "32-true")


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.
    
    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r determines the scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    
    def to_dict(self):
        """Convert to dictionary for PEFT config."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


@dataclass
class LoKRConfig:
    """Configuration for LoKr (Low-Rank Kronecker) training via LyCORIS.

    LoKr uses Kronecker product factorization as an alternative to standard
    LoRA low-rank decomposition. Can offer different quality/compression
    trade-offs depending on the model architecture.

    Requires: pip install lycoris-lora

    Attributes:
        linear_dim: Dimension for the linear component (similar to LoRA rank).
                     10000 = let factor determine the effective rank automatically.
        linear_alpha: Scaling factor (similar to LoRA alpha). 1.0 recommended.
        factor: Kronecker factor (-1 = auto sqrt of dimension, positive = fixed).
                Controls the decomposition granularity.
        decompose_both: Decompose both weight matrices in the Kronecker product.
                         Can improve quality at cost of more parameters.
        use_tucker: Use Tucker decomposition for additional compression.
        dropout: Dropout probability for LoKr layers.
        target_modules: List of module names to apply LoKr to.
    """
    linear_dim: int = 10000  # 10000 = auto (factor determines effective rank)
    linear_alpha: float = 1.0
    factor: int = -1  # -1 = auto (sqrt of dimension)
    decompose_both: bool = False
    use_tucker: bool = False
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "linear_dim": self.linear_dim,
            "linear_alpha": self.linear_alpha,
            "factor": self.factor,
            "decompose_both": self.decompose_both,
            "use_tucker": self.use_tucker,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


@dataclass
class TrainingConfig:
    """Configuration for LoRA training process.

    Supports two model types:
    - **turbo** (default): Distilled model, 8 discrete timesteps, shift=3.0, no CFG
    - **base**: Full diffusion model, continuous timesteps, configurable shift, CFG supported

    Attributes:
        model_type: "turbo" or "base" - determines timestep schedule and CFG
        shift: Timestep shift factor (3.0 for turbo, 1.0-5.0 for base)
        num_inference_steps: Number of steps (8 for turbo, 32-100 for base)
        guidance_scale: CFG scale (ignored for turbo, 5.0-15.0 for base, 0=disabled)
        cfg_dropout_prob: Probability of dropping condition for CFG training (base only)
        learning_rate: Initial learning rate
        batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_epochs: Maximum number of training epochs
        save_every_n_epochs: Save checkpoint every N epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        mixed_precision: "auto" (detect), "bf16", "fp16", or "fp32"
        seed: Random seed for reproducibility
        output_dir: Directory to save checkpoints and logs
    """
    # Adapter type: "lora" (PEFT, default) or "lokr" (LyCORIS Kronecker)
    adapter_type: str = "lora"

    # Model type: "turbo" (8 discrete steps, no CFG) or "base" (continuous, with CFG)
    model_type: str = "turbo"

    # Timestep / schedule settings
    shift: float = 3.0  # Turbo: fixed 3.0. Base: adjustable 1.0-5.0
    num_inference_steps: int = 8  # Turbo: 8. Base: 32-100

    # Dynamic shift (turbo only): sample shift from {1,2,3} per batch
    # As described in ACE-Step paper: "dynamic-shift strategy with the shift
    # parameter stochastically sampled from {1,2,3}"
    dynamic_shift: bool = False

    # CFG settings
    guidance_scale: float = 0.0  # 0 = disabled. Base: 5.0-15.0 recommended
    cfg_dropout_prob: float = 0.15  # Probability of dropping condition during training (all model types)

    # Timestep sampling (logit-normal distribution matching pre-training)
    # t = sigmoid(N(mu, sigma)) — these defaults match the model's own sample_t_r()
    timestep_mu: float = -0.4  # Logit-normal mean (negative = bias toward cleaner data)
    timestep_sigma: float = 1.0  # Logit-normal standard deviation

    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    save_every_n_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "auto"  # "auto", "bf16", "fp16", or "fp32"
    seed: int = 42
    output_dir: str = "./lora_output"

    # MLP targeting: "true" (default) or "false"
    # When enabled, targets MLP modules (gate_proj, up_proj, down_proj) in addition to attention.
    # Significantly increases parameters and quality, but uses more VRAM.
    train_mlp: bool = False

    # Optimizer: "adamw" (default), "adamw8bit" (bitsandbytes), "adafactor", "prodigy"
    optimizer_type: str = "adamw"

    # Scheduler: "cosine" (default), "cosine_restarts", "linear", "constant", "constant_with_warmup"
    scheduler_type: str = "cosine"

    # Attention targeting: "both" (default), "self", "cross"
    # - "self": only self-attention layers (self_attn.q/k/v/o_proj)
    # - "cross": only cross-attention layers (cross_attn.q/k/v/o_proj)
    # - "both": all attention layers
    attention_type: str = "both"

    # Gradient checkpointing: saves ~40-60% activation VRAM at ~30% speed cost.
    # Recommended for GPUs with ≤16 GB VRAM.
    gradient_checkpointing: bool = False

    # Encoder offloading: moves non-decoder components (encoder, VAE) to CPU
    # during training. Saves ~2-4 GB VRAM. Only relevant if service is loaded;
    # preprocessed tensor training already doesn't use the encoder on GPU.
    encoder_offloading: bool = False

    # torch.compile: JIT-compile the decoder for potential speedup.
    # Default OFF — on Windows the first epoch takes minutes of JIT warm-up,
    # and variable-length tensors cause repeated recompilation.
    # Enable only on Linux with fixed-length audio for best results.
    torch_compile: bool = False

    # Max latent time-steps per sample during training.
    # 0 = use full length (can be very slow for long audio).
    # 1500 ≈ 60s, 3000 ≈ 120s.  Shorter sequences train dramatically faster
    # because DiT self-attention is O(T²).  A random crop window is chosen
    # each epoch, providing implicit data augmentation.
    # Default 1500 matches ACE-Step's standard 60s inference length.
    max_latent_length: int = 1500

    # Data loading (num_workers=0 on Windows avoids spawn overhead for small .pt files)
    num_workers: int = 0
    pin_memory: bool = True

    # Early stopping (0 = disabled, >0 = patience in epochs)
    early_stop_patience: int = 0

    # Auto-save best model: after this many warmup epochs, automatically
    # save the best checkpoint (lowest loss) to output_dir/best/.
    # 0 = disabled.  Default 200 — gives the model enough time to produce
    # quality results before we start tracking the best weights.
    auto_save_best_after: int = 200

    # ── Sample inference during training ──
    # When enabled, generates test audio at regular epoch intervals so
    # you can hear how the model evolves (loss alone can't tell you quality).
    sample_enabled: bool = False
    sample_every_n_epochs: int = 0  # 0 = same as save_every_n_epochs
    sample_prompt: str = ""  # Style/genre description
    sample_lyrics: str = ""  # Optional lyrics
    sample_bpm: int = 120
    sample_key: str = ""  # e.g., "D Minor"
    sample_time_signature: str = "4"
    sample_duration: float = 30.0  # seconds (shorter = faster + less VRAM)
    sample_strengths: str = "1.0"  # comma-separated LoRA strengths
    sample_inference_steps: int = 0  # 0 = use training config's num_inference_steps
    sample_guidance_scale: float = 0.0  # 0 = use training config's guidance_scale
    sample_shift: float = 0.0  # 0 = use training config's shift
    sample_seed: int = 42  # fixed seed for consistency across epochs

    # ── Loss weighting ──
    # "none" = standard unweighted MSE (default)
    # "min_snr" = Min-SNR-gamma weighting (from "Efficient Diffusion Training
    #   via Min-SNR Weighting Strategy", Hang et al. 2023).
    #   Weights per-timestep loss by min(SNR, gamma) / SNR, reducing the
    #   contribution of easy-to-predict (high-SNR) timesteps while clamping
    #   the weight at gamma to avoid exploding gradients for noisy timesteps.
    loss_weighting: str = "none"  # "none" or "min_snr"
    snr_gamma: float = 5.0  # Clamping value for Min-SNR (5.0 recommended)

    # ── NaN / Inf detection ──
    # Automatically detect NaN or Inf losses during training and halt if
    # too many consecutive bad batches are encountered.
    # 0 = disabled.  10 = halt after 10 consecutive NaN/Inf losses.
    nan_detection_max: int = 10

    # ── Audio normalization during preprocessing ──
    # "none" = no normalization (default, raw audio as-is)
    # "peak" = peak-normalize to -1.0 dBFS (loudest sample = 1.0)
    # "lufs" = loudness-normalize to -14 LUFS (broadcast standard)
    # "peak_lufs" = both: peak-normalize first, then LUFS-normalize
    audio_normalization: str = "none"

    # Logging
    log_every_n_steps: int = 10

    @property
    def is_turbo(self) -> bool:
        """Check if training is for turbo model."""
        return self.model_type == "turbo"

    @property
    def use_cfg(self) -> bool:
        """Check if CFG dropout is enabled.

        CFG dropout is now applied to ALL model types (turbo included)
        when cfg_dropout_prob > 0. This matches the original pre-training
        which used null_condition_emb for unconditional generation.
        """
        return self.cfg_dropout_prob > 0
    
    def resolve_precision(self) -> str:
        """Resolve the actual precision to use.

        If mixed_precision is "auto", detect the best precision for the device.
        Otherwise returns the configured value.

        Returns:
            Resolved precision string: "bf16", "fp16", or "fp32"
        """
        if self.mixed_precision == "auto":
            resolved = detect_best_precision()
            logger.info(f"Auto-detected training precision: {resolved}")
            return resolved
        return self.mixed_precision

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "adapter_type": self.adapter_type,
            "model_type": self.model_type,
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
            "dynamic_shift": self.dynamic_shift,
            "guidance_scale": self.guidance_scale,
            "cfg_dropout_prob": self.cfg_dropout_prob,
            "timestep_mu": self.timestep_mu,
            "timestep_sigma": self.timestep_sigma,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "save_every_n_epochs": self.save_every_n_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "optimizer_type": self.optimizer_type,
            "scheduler_type": self.scheduler_type,
            "attention_type": self.attention_type,
            "gradient_checkpointing": self.gradient_checkpointing,
            "encoder_offloading": self.encoder_offloading,
            "torch_compile": self.torch_compile,
            "max_latent_length": self.max_latent_length,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "log_every_n_steps": self.log_every_n_steps,
            "early_stop_patience": self.early_stop_patience,
            "auto_save_best_after": self.auto_save_best_after,
            "loss_weighting": self.loss_weighting,
            "snr_gamma": self.snr_gamma,
            "nan_detection_max": self.nan_detection_max,
            "audio_normalization": self.audio_normalization,
            "train_mlp": self.train_mlp,
        }


# ============================================================================
# VRAM Preset System
# ============================================================================

def list_presets() -> List[str]:
    """List all available training preset names.

    Scans the presets/ directory for JSON files.

    Returns:
        List of preset names (without .json extension), sorted alphabetically.
    """
    if not os.path.isdir(_PRESETS_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(_PRESETS_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )


def load_preset(name: str) -> Dict[str, Any]:
    """Load a training preset by name.

    Args:
        name: Preset name (e.g., "recommended", "vram_8gb").
              Can also be an absolute path to a custom JSON file.

    Returns:
        Dictionary of preset parameters.

    Raises:
        FileNotFoundError: If preset not found.
    """
    # Allow absolute/relative paths to custom presets
    if os.path.isfile(name):
        path = name
    else:
        path = os.path.join(_PRESETS_DIR, f"{name}.json")

    if not os.path.isfile(path):
        available = list_presets()
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available presets: {', '.join(available)}"
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Remove internal keys
    data.pop("_description", None)

    logger.info(f"Loaded training preset: {name}")
    return data


def apply_preset(
    training_config: "TrainingConfig",
    lora_config: "LoRAConfig",
    preset_name: str,
) -> str:
    """Apply a preset to existing config objects.

    Updates TrainingConfig and LoRAConfig in-place with values from the preset.
    Only overwrites fields that are present in the preset JSON; other fields
    retain their current values.

    Args:
        training_config: TrainingConfig to update.
        lora_config: LoRAConfig to update.
        preset_name: Name of the preset to apply.

    Returns:
        Description string from the preset.
    """
    # Load the raw preset (may contain _description)
    if os.path.isfile(preset_name):
        path = preset_name
    else:
        path = os.path.join(_PRESETS_DIR, f"{preset_name}.json")

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    description = raw.get("_description", f"Preset: {preset_name}")
    data = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Map lora_* keys to LoRAConfig fields
    lora_mapping = {
        "lora_r": "r",
        "lora_alpha": "alpha",
        "lora_dropout": "dropout",
    }

    for key, value in data.items():
        if key in lora_mapping:
            setattr(lora_config, lora_mapping[key], value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)

    logger.info(f"Applied preset '{preset_name}': {description}")
    return description


def auto_select_preset() -> str:
    """Auto-select the best preset based on available GPU VRAM.

    Returns:
        Preset name string.
    """
    try:
        if torch.cuda.is_available():
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            vram_gb = total_mb / 1024
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            total_mb = torch.xpu.get_device_properties(0).total_memory / (1024 ** 2)
            vram_gb = total_mb / 1024
        else:
            vram_gb = 0
    except Exception:
        vram_gb = 0

    if vram_gb >= 24:
        return "vram_24gb_plus"
    elif vram_gb >= 16:
        return "vram_16gb"
    elif vram_gb >= 12:
        return "vram_12gb"
    elif vram_gb >= 8:
        return "vram_8gb"
    else:
        return "vram_8gb"  # Fallback to most conservative
