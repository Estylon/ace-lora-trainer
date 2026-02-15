"""
LoRA Trainer for ACE-Step

Lightning Fabric-based trainer for LoRA fine-tuning of ACE-Step DiT decoder.
Supports training from preprocessed tensor files for optimal performance.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple, Generator
from loguru import logger

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
    SequentialLR,
)

from contextlib import nullcontext as _nullcontext

try:
    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning Fabric not installed. Training will use basic training loop.")

from acestep.training.configs import LoRAConfig, TrainingConfig, precision_to_dtype, precision_to_fabric_str
from acestep.training.lora_utils import (
    inject_lora_into_dit,
    save_lora_weights,
    save_training_checkpoint,
    load_training_checkpoint,
    check_peft_available,
    check_model_quantized,
    upcast_model_for_training,
)
from acestep.training.data_module import PreprocessedDataModule

# Optional: LyCORIS for LoKr support
try:
    from acestep.training.lokr_utils import (
        inject_lokr_into_dit,
        save_lokr_weights,
        save_lokr_training_checkpoint,
        check_lycoris_available,
    )
    LOKR_IMPORTS_OK = True
except ImportError:
    LOKR_IMPORTS_OK = False


def create_optimizer(params, training_config) -> torch.optim.Optimizer:
    """Create optimizer based on training config.

    Supports:
    - adamw: Standard AdamW (default). Uses fused=True on CUDA for speed.
    - adamw8bit: 8-bit AdamW via bitsandbytes (~30-40% less optimizer VRAM).
                 Falls back to standard AdamW if bitsandbytes not installed.
    - adafactor: Memory-efficient optimizer from HuggingFace transformers.
    - prodigy: Auto-learning-rate optimizer. Overrides LR to 1.0 as recommended.
    """
    opt_type = training_config.optimizer_type.lower()
    lr = training_config.learning_rate
    wd = training_config.weight_decay

    if opt_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
            logger.info("Using AdamW 8-bit optimizer (bitsandbytes)")
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd)
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to standard AdamW")
            opt_type = "adamw"

    if opt_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            logger.info("Using Adafactor optimizer")
            return Adafactor(
                params, lr=lr, weight_decay=wd,
                scale_parameter=False, relative_step=False,
            )
        except ImportError:
            logger.warning("transformers not installed, falling back to standard AdamW")
            opt_type = "adamw"

    if opt_type == "prodigy":
        try:
            from prodigyopt import Prodigy
            # Prodigy manages its own LR. Recommended to set lr=1.0.
            # If user left the default 1e-4, override to 1.0.
            prodigy_lr = 1.0 if lr <= 1e-3 else lr
            logger.info(f"Using Prodigy optimizer (lr={prodigy_lr})")
            return Prodigy(params, lr=prodigy_lr, weight_decay=wd)
        except ImportError:
            logger.warning("prodigyopt not installed, falling back to standard AdamW. "
                           "Install with: pip install prodigyopt")
            opt_type = "adamw"

    # Default: standard AdamW
    use_fused = torch.cuda.is_available()
    logger.info(f"Using AdamW optimizer (fused={use_fused})")
    return AdamW(params, lr=lr, weight_decay=wd, fused=use_fused)


def create_scheduler(optimizer, training_config, total_steps: int):
    """Create learning rate scheduler based on training config.

    All schedulers include a linear warmup phase.
    For Prodigy optimizer, forces constant scheduler since Prodigy manages LR internally.

    Supports:
    - cosine: Cosine annealing to 1% of initial LR (default)
    - linear: Linear decay to 1% of initial LR
    - constant: Constant LR (no decay after warmup)
    - constant_with_warmup: Same as constant (warmup is always included)
    """
    sched_type = training_config.scheduler_type.lower()
    lr = training_config.learning_rate
    warmup_steps = min(training_config.warmup_steps, max(1, total_steps // 10))

    # Prodigy manages its own LR — force constant scheduler
    if training_config.optimizer_type.lower() == "prodigy" and sched_type not in ("constant", "constant_with_warmup"):
        logger.info(f"Prodigy optimizer detected: overriding scheduler from '{sched_type}' to 'constant'")
        sched_type = "constant"

    # Warmup phase (always present)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    main_steps = max(1, total_steps - warmup_steps)

    if sched_type == "linear":
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=main_steps,
        )
    elif sched_type in ("constant", "constant_with_warmup"):
        main_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=main_steps,
        )
    else:
        # Default: cosine
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=main_steps,
            eta_min=lr * 0.01,
        )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    logger.info(f"Scheduler: {sched_type} (warmup={warmup_steps} steps, total={total_steps} steps)")
    return scheduler


def sample_logit_normal_timestep(bsz, device, dtype, mu=-0.4, sigma=1.0):
    """Sample timesteps from logit-normal distribution matching pre-training.

    This is a faithful reproduction of the model's own sample_t_r() function
    from modeling_acestep_v15_turbo.py. The logit-normal distribution:
        t = sigmoid(N(mu, sigma))
    creates a bell-shaped distribution in [0,1] that matches what the model
    saw during pre-training. This is critical for LoRA training quality.

    The old discrete 8-step schedule (used in v1) was designed for inference,
    not training, and caused a distribution mismatch that degraded LoRA quality.

    Args:
        bsz: Batch size
        device: Device
        dtype: Data type
        mu: Logit-normal mean (-0.4 = bias toward cleaner data, matching pre-training)
        sigma: Logit-normal standard deviation (1.0 = moderate spread)

    Returns:
        Tuple of (t, r) — for current ACE-Step variants, r == t
    """
    # Sample from logit-normal: t = sigmoid(N(mu, sigma))
    t = torch.sigmoid(torch.randn((bsz,), device=device, dtype=dtype) * sigma + mu)
    r = torch.sigmoid(torch.randn((bsz,), device=device, dtype=dtype) * sigma + mu)

    # Ensure t >= r (t is the noisier timestep)
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # For current ACE-Step variants, data_proportion = 1.0 so r = t
    # (infrastructure for future mean-flow support is preserved above)
    r = t

    return t, r


def apply_cfg_dropout(encoder_hidden_states, null_condition_emb, cfg_ratio=0.15):
    """Apply classifier-free guidance dropout using model's learned null embedding.

    Replaces randomly selected condition embeddings with the model's own
    null_condition_emb (the learned unconditional embedding from pre-training).

    The v1 approach of zeroing out embeddings was incorrect — the model was
    pre-trained with a specific learned null embedding, not zeros. Using zeros
    produces incorrect gradient signals during LoRA training.

    Args:
        encoder_hidden_states: [B, L, D] condition embeddings
        null_condition_emb: The model's learned null/unconditional embedding
        cfg_ratio: Probability of replacing each sample's condition (default 0.15)

    Returns:
        Modified encoder_hidden_states with some samples replaced by null embedding
    """
    if cfg_ratio <= 0:
        return encoder_hidden_states

    bsz = encoder_hidden_states.shape[0]
    drop_mask = torch.rand(bsz, device=encoder_hidden_states.device) < cfg_ratio

    if not drop_mask.any():
        return encoder_hidden_states

    # Clone to avoid in-place modification
    encoder_hidden_states = encoder_hidden_states.clone()

    # Replace dropped samples with the model's null condition embedding
    # null_condition_emb shape may be [1, L, D] or [L, D] — broadcast to match
    null_emb = null_condition_emb.to(
        device=encoder_hidden_states.device,
        dtype=encoder_hidden_states.dtype,
    )
    if null_emb.dim() == 2:
        null_emb = null_emb.unsqueeze(0)

    # Expand null_emb to match the sequence length of encoder_hidden_states
    seq_len = encoder_hidden_states.shape[1]
    null_seq_len = null_emb.shape[1]
    if null_seq_len < seq_len:
        # Pad null_emb with zeros to match
        padding = torch.zeros(
            1, seq_len - null_seq_len, null_emb.shape[2],
            device=null_emb.device, dtype=null_emb.dtype,
        )
        null_emb = torch.cat([null_emb, padding], dim=1)
    elif null_seq_len > seq_len:
        null_emb = null_emb[:, :seq_len, :]

    encoder_hidden_states[drop_mask] = null_emb.expand(drop_mask.sum(), -1, -1)

    return encoder_hidden_states


class PreprocessedLoRAModule(nn.Module):
    """LoRA Training Module using preprocessed tensors.
    
    This module trains only the DiT decoder with LoRA adapters.
    All inputs are pre-computed tensors - no VAE or text encoder needed!
    
    Training flow:
    1. Load pre-computed tensors (target_latents, encoder_hidden_states, context_latents)
    2. Sample noise and timestep
    3. Forward through decoder (with LoRA)
    4. Compute flow matching loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype,
        lokr_config=None,
    ):
        """Initialize the training module.

        Args:
            model: The AceStepConditionGenerationModel
            lora_config: LoRA configuration
            training_config: Training configuration
            device: Device to use
            dtype: Data type to use
            lokr_config: Optional LoKRConfig for LoKr training
        """
        super().__init__()

        self.lora_config = lora_config
        self.training_config = training_config
        self.device = device
        self.dtype = dtype

        # Determine adapter type
        adapter_type = getattr(training_config, 'adapter_type', 'lora')
        self.adapter_type = adapter_type
        self.lycoris_net = None  # Only set for LoKr

        # Inject adapter into the decoder
        attention_type = getattr(training_config, 'attention_type', 'both')

        if adapter_type == "lokr":
            if not LOKR_IMPORTS_OK or not check_lycoris_available():
                raise ImportError(
                    "LyCORIS library is required for LoKr training. "
                    "Install with: pip install lycoris-lora"
                )
            if lokr_config is None:
                raise ValueError("lokr_config is required when adapter_type='lokr'")
            self.model, self.lycoris_net, self.lora_info = inject_lokr_into_dit(
                model, lokr_config, attention_type=attention_type,
            )
            logger.info(f"LoKr injected: {self.lora_info['trainable_params']:,} trainable params (attention: {attention_type})")
        elif check_peft_available():
            self.model, self.lora_info = inject_lora_into_dit(model, lora_config, attention_type=attention_type)
            logger.info(f"LoRA injected: {self.lora_info['trainable_params']:,} trainable params (attention: {attention_type})")
        else:
            self.model = model
            self.lora_info = {}
            logger.warning("No adapter library available, training without adapters")

        # Model config for flow matching
        self.config = model.config

        # Cache the model's null_condition_emb for CFG dropout
        # This is the learned unconditional embedding from pre-training
        self._null_condition_emb = getattr(model, 'null_condition_emb', None)
        if self._null_condition_emb is None:
            # Try to find it on the decoder
            self._null_condition_emb = getattr(model.decoder, 'null_condition_emb', None)
        if self._null_condition_emb is not None:
            logger.info("Found model.null_condition_emb for CFG dropout")
        else:
            logger.warning("model.null_condition_emb not found — CFG dropout will zero embeddings instead")

        # Store training losses
        self.training_losses = []
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step using preprocessed tensors.

        Supports both turbo (discrete timesteps, no CFG) and base (continuous
        timesteps, optional CFG) model training.

        Args:
            batch: Dictionary containing pre-computed tensors:
                - target_latents: [B, T, 64] - VAE encoded audio
                - attention_mask: [B, T] - Valid audio mask
                - encoder_hidden_states: [B, L, D] - Condition encoder output
                - encoder_attention_mask: [B, L] - Condition mask
                - context_latents: [B, T, 128] - Source context

        Returns:
            Loss tensor (float32 for stable backward)
        """
        # Determine autocast context based on device
        # self.device may be a torch.device object or a plain string (e.g. from Fabric)
        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]
        use_autocast = device_type in ("cuda", "xpu") and self.dtype != torch.float32
        autocast_ctx = torch.autocast(device_type=device_type, dtype=self.dtype) if use_autocast else _nullcontext()

        with autocast_ctx:
            # Tensors are already on device from Fabric's setup_dataloaders
            target_latents = batch["target_latents"]  # x0
            attention_mask = batch["attention_mask"]
            encoder_hidden_states = batch["encoder_hidden_states"]
            encoder_attention_mask = batch["encoder_attention_mask"]
            context_latents = batch["context_latents"]

            bsz = target_latents.shape[0]

            # Flow matching: sample noise x1 and interpolate with data x0
            x1 = torch.randn_like(target_latents)  # Noise
            x0 = target_latents  # Data

            # Logit-normal timestep sampling (matches model pre-training distribution)
            # This replaces the old discrete 8-step schedule (turbo) and uniform+shift (base)
            t, r = sample_logit_normal_timestep(
                bsz, self.device, self.dtype,
                mu=self.training_config.timestep_mu,
                sigma=self.training_config.timestep_sigma,
            )

            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # Interpolate: x_t = t * x1 + (1 - t) * x0
            xt = t_ * x1 + (1.0 - t_) * x0

            # CFG dropout: replace random samples' condition with null embedding
            # Applied to ALL model types (turbo + base) matching pre-training
            if self.training_config.use_cfg and self._null_condition_emb is not None:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    self._null_condition_emb,
                    cfg_ratio=self.training_config.cfg_dropout_prob,
                )
            elif self.training_config.use_cfg:
                # Fallback: zero out embeddings if null_condition_emb not available
                drop_mask = torch.rand(bsz, device=self.device) < self.training_config.cfg_dropout_prob
                if drop_mask.any():
                    drop_mask_expanded = drop_mask.unsqueeze(-1).unsqueeze(-1)
                    encoder_hidden_states = encoder_hidden_states * (~drop_mask_expanded).float()
                    encoder_attention_mask = encoder_attention_mask * (~drop_mask.unsqueeze(-1)).float()

            # Forward through decoder
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=r,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # Flow matching loss: predict the flow field v = x1 - x0
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        # Convert loss to float32 for stable backward pass
        diffusion_loss = diffusion_loss.float()

        self.training_losses.append(diffusion_loss.item())
        # Keep only last 1000 losses to avoid unbounded memory growth
        if len(self.training_losses) > 1000:
            self.training_losses = self.training_losses[-500:]

        return diffusion_loss


class LoRATrainer:
    """High-level trainer for ACE-Step LoRA fine-tuning.
    
    Uses Lightning Fabric for distributed training and mixed precision.
    Supports training from preprocessed tensor directories.
    """
    
    def __init__(
        self,
        dit_handler,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        lokr_config=None,
    ):
        """Initialize the trainer.

        Args:
            dit_handler: Initialized DiT handler (for model access)
            lora_config: LoRA configuration
            training_config: Training configuration
            lokr_config: Optional LoKRConfig for LoKr training
        """
        self.dit_handler = dit_handler
        self.lora_config = lora_config
        self.training_config = training_config
        self.lokr_config = lokr_config

        self.module = None
        self.fabric = None
        self.is_training = False
    
    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train LoRA adapters from preprocessed tensor files.

        This is the recommended training method for best performance.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            training_state: Optional state dict for stopping control
            resume_from: Optional path to checkpoint directory to resume from

        Yields:
            Tuples of (step, loss, status_message)
        """
        self.is_training = True
        
        try:
            # Check for quantization before anything else
            quant_type = check_model_quantized(self.dit_handler.model)
            if quant_type != "none":
                if quant_type in ("bitsandbytes", "int_weights"):
                    yield 0, 0.0, f"❌ Cannot train LoRA on a {quant_type}-quantized model. Please restart without quantization."
                    return
                # For fp8/torchao: warn and upcast
                resolved_prec = self.training_config.resolve_precision()
                upcast_dtype = precision_to_dtype(resolved_prec)
                upcast_msg = upcast_model_for_training(self.dit_handler.model, upcast_dtype)
                yield 0, 0.0, f"⚠️ Quantized model ({quant_type}) detected. {upcast_msg}"

            # Validate tensor directory
            if not os.path.exists(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return
            
            # Create training module
            self.module = PreprocessedLoRAModule(
                model=self.dit_handler.model,
                lora_config=self.lora_config,
                training_config=self.training_config,
                device=self.dit_handler.device,
                dtype=self.dit_handler.dtype,
                lokr_config=self.lokr_config,
            )
            
            # Create data module
            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                max_latent_length=self.training_config.max_latent_length,
            )
            
            # Setup data
            data_module.setup('fit')
            
            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return
            
            crop_info = ""
            if self.training_config.max_latent_length > 0:
                crop_secs = self.training_config.max_latent_length / 25  # 25 frames/sec
                crop_info = f" (random crop to ~{crop_secs:.0f}s per sample)"
            yield 0, 0.0, f"📂 Loaded {len(data_module.train_dataset)} preprocessed samples{crop_info}"

            if LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(data_module, training_state, resume_from)
            else:
                yield from self._train_basic(data_module, training_state, resume_from)
                
        except Exception as e:
            logger.exception("Training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False
    
    def _save_adapter_weights(self, output_dir: str):
        """Save adapter weights (LoRA or LoKr) to output_dir."""
        if self.module.adapter_type == "lokr" and self.module.lycoris_net is not None:
            save_lokr_weights(
                self.module.lycoris_net, output_dir,
                lokr_config=self.lokr_config,
            )
        else:
            save_lora_weights(self.module.model, output_dir)

    def _save_adapter_checkpoint(self, optimizer, scheduler, epoch, global_step, output_dir):
        """Save training checkpoint (LoRA or LoKr) to output_dir."""
        if self.module.adapter_type == "lokr" and self.module.lycoris_net is not None:
            save_lokr_training_checkpoint(
                self.module.lycoris_net, optimizer, scheduler,
                epoch, global_step, output_dir,
                lokr_config=self.lokr_config,
            )
        else:
            save_training_checkpoint(
                self.module.model, optimizer, scheduler,
                epoch, global_step, output_dir,
            )

    def _train_with_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train using Lightning Fabric."""
        # Create output directory
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # Resolve precision: auto-detect or use configured value
        resolved_precision = self.training_config.resolve_precision()
        precision = precision_to_fabric_str(resolved_precision)
        train_dtype = precision_to_dtype(resolved_precision)

        # Create TensorBoard logger (optional — tensorboard may not be installed)
        fabric_loggers = []
        try:
            tb_logger = TensorBoardLogger(
                root_dir=self.training_config.output_dir,
                name="logs"
            )
            fabric_loggers.append(tb_logger)
        except Exception as e:
            logger.warning(f"TensorBoard logger not available ({e}), training will proceed without it")

        # Initialize Fabric
        self.fabric = Fabric(
            accelerator="auto",
            devices=1,
            precision=precision,
            loggers=fabric_loggers,
        )
        self.fabric.launch()
        
        model_info = f"{'turbo' if self.training_config.is_turbo else 'base'}"
        model_info += f", timestep: logit-normal(μ={self.training_config.timestep_mu}, σ={self.training_config.timestep_sigma})"
        if self.training_config.use_cfg:
            model_info += f", CFG dropout={self.training_config.cfg_dropout_prob:.0%}"
        yield 0, 0.0, f"🚀 Starting training ({model_info}, precision: {resolved_precision})..."
        
        # Get dataloader
        train_loader = data_module.train_dataloader()

        # Setup optimizer — get trainable params from the right source
        if self.module.adapter_type == "lokr" and self.module.lycoris_net is not None:
            trainable_params = list(self.module.lycoris_net.parameters())
        else:
            trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        adapter_label = "LoKr" if self.module.adapter_type == "lokr" else "LoRA"
        yield 0, 0.0, f"🎯 Training {sum(p.numel() for p in trainable_params):,} {adapter_label} parameters"

        optimizer = create_optimizer(trainable_params, self.training_config)

        # Calculate total steps
        total_steps = len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps

        scheduler = create_scheduler(optimizer, self.training_config, total_steps)

        yield 0, 0.0, f"📋 Optimizer: {self.training_config.optimizer_type}, Scheduler: {self.training_config.scheduler_type}"

        # Encoder offloading: move non-decoder components to CPU to save VRAM
        if self.training_config.encoder_offloading:
            try:
                offloaded = []
                for attr_name in ('encoder', 'text_encoder', 'vae', 'tokenizer'):
                    component = getattr(self.dit_handler.model, attr_name, None)
                    if component is not None and hasattr(component, 'to'):
                        try:
                            component.to('cpu')
                            offloaded.append(attr_name)
                        except Exception:
                            pass
                if offloaded:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    yield 0, 0.0, f"📤 Offloaded to CPU: {', '.join(offloaded)} (VRAM saved)"
            except Exception as e:
                logger.warning(f"Encoder offloading failed: {e}")

        # Convert model to training dtype (auto-detected or configured)
        self.module.dtype = train_dtype
        self.module.model = self.module.model.to(train_dtype)

        # Setup with Fabric
        # For LoRA: setup the decoder (which contains PEFT wrapper)
        # For LoKr: setup BOTH lycoris_net AND the decoder with Fabric.
        #   The decoder must go through Fabric for proper mixed-precision handling.
        #   LyCORIS hooks modify the forward pass, but Fabric needs to manage the
        #   decoder's autocast/grad-scaling. We also ensure the lycoris_net params
        #   are on the correct device and dtype before Fabric wraps them.
        if self.module.adapter_type == "lokr" and self.module.lycoris_net is not None:
            # Ensure lycoris_net params match the training dtype and device
            target_device = self.module.device
            self.module.lycoris_net = self.module.lycoris_net.to(device=target_device, dtype=train_dtype)
            # Ensure all decoder params (including any LyCORIS-injected ones) are co-located
            self.module.model.decoder = self.module.model.decoder.to(device=target_device)
            self.module.lycoris_net, optimizer = self.fabric.setup(self.module.lycoris_net, optimizer)
        else:
            self.module.model.decoder, optimizer = self.fabric.setup(self.module.model.decoder, optimizer)

        # Compile decoder for faster training (after Fabric setup)
        # Off by default: JIT warm-up is very slow on Windows, and variable-length
        # tensors cause repeated recompilation that makes training *slower*.
        if self.training_config.torch_compile:
            try:
                import sys
                compile_mode = "default" if sys.platform == "win32" else "reduce-overhead"
                self.module.model.decoder = torch.compile(
                    self.module.model.decoder,
                    mode=compile_mode,
                    dynamic=True,  # allow variable-length tensors without full recompile
                )
                yield 0, 0.0, f"⚡ torch.compile enabled (mode={compile_mode}, dynamic=True)"
            except Exception as e:
                logger.warning(f"torch.compile not available, skipping: {e}")
                yield 0, 0.0, f"⚠️ torch.compile skipped, continuing without"
        else:
            yield 0, 0.0, "⏩ torch.compile disabled (faster start, recommended for small datasets)"

        # Gradient checkpointing: trade speed for VRAM by recomputing activations
        if self.training_config.gradient_checkpointing:
            try:
                decoder = self.module.model.decoder
                # Unwrap Fabric wrapper if present
                raw_decoder = decoder._forward_module if hasattr(decoder, '_forward_module') else decoder
                # Unwrap PEFT wrapper if present
                if hasattr(raw_decoder, 'base_model') and hasattr(raw_decoder.base_model, 'model'):
                    raw_decoder = raw_decoder.base_model.model
                if hasattr(raw_decoder, 'gradient_checkpointing_enable'):
                    raw_decoder.gradient_checkpointing_enable()
                    yield 0, 0.0, "♻️ Gradient checkpointing enabled (saves VRAM, ~30% slower)"
                else:
                    # Manual fallback: enable on individual transformer blocks
                    enabled_count = 0
                    for module in raw_decoder.modules():
                        if hasattr(module, 'gradient_checkpointing_enable'):
                            module.gradient_checkpointing_enable()
                            enabled_count += 1
                    if enabled_count > 0:
                        yield 0, 0.0, f"♻️ Gradient checkpointing enabled on {enabled_count} blocks"
                    else:
                        yield 0, 0.0, "⚠️ Gradient checkpointing not supported by this model"
            except Exception as e:
                logger.warning(f"Gradient checkpointing failed: {e}")
                yield 0, 0.0, f"⚠️ Gradient checkpointing failed: {e}"

        train_loader = self.fabric.setup_dataloaders(train_loader)

        # Handle resume from checkpoint (load AFTER Fabric setup)
        start_epoch = 0
        global_step = 0
        checkpoint_info = None

        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."

                # Load checkpoint using utility function
                checkpoint_info = load_training_checkpoint(
                    resume_from,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                )

                if checkpoint_info["adapter_path"]:
                    adapter_path = checkpoint_info["adapter_path"]
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                    if not os.path.exists(adapter_weights_path):
                        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")

                    if os.path.exists(adapter_weights_path):
                        # Load adapter weights
                        from safetensors.torch import load_file
                        if adapter_weights_path.endswith(".safetensors"):
                            state_dict = load_file(adapter_weights_path)
                        else:
                            state_dict = torch.load(adapter_weights_path, map_location=self.module.device)

                        # Get the decoder (might be wrapped by Fabric)
                        decoder = self.module.model.decoder
                        if hasattr(decoder, '_forward_module'):
                            decoder = decoder._forward_module

                        decoder.load_state_dict(state_dict, strict=False)

                        start_epoch = checkpoint_info["epoch"]
                        global_step = checkpoint_info["global_step"]

                        status_parts = [f"✅ Resumed from epoch {start_epoch}, step {global_step}"]
                        if checkpoint_info["loaded_optimizer"]:
                            status_parts.append("optimizer ✓")
                        if checkpoint_info["loaded_scheduler"]:
                            status_parts.append("scheduler ✓")
                        yield 0, 0.0, ", ".join(status_parts)
                    else:
                        yield 0, 0.0, f"⚠️ Adapter weights not found in {adapter_path}"
                else:
                    yield 0, 0.0, f"⚠️ No valid checkpoint found in {resume_from}"

            except Exception as e:
                logger.exception("Failed to load checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
                start_epoch = 0
                global_step = 0
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        # Training loop
        accumulation_step = 0
        accumulated_loss = 0.0

        # Best loss tracking — only counts after auto_save_best_after warmup.
        # Uses a moving average (window=5) instead of single-epoch loss to
        # avoid saving on random low-loss spikes.
        best_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_tracking_active = False  # flips True once we pass the warmup
        early_stop_patience = getattr(self.training_config, 'early_stop_patience', 0)
        auto_save_best_after = getattr(self.training_config, 'auto_save_best_after', 200)
        min_delta = 0.001
        loss_window_size = 5
        recent_losses = []  # rolling window for smoothed best-loss tracking

        self.module.model.decoder.train()

        # Determine which module Fabric manages for gradient clipping
        clip_module = self.module.lycoris_net if (self.module.adapter_type == "lokr" and self.module.lycoris_net is not None) else self.module.model.decoder

        for epoch in range(start_epoch, self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # Check for stop signal
                if training_state and training_state.get("should_stop", False):
                    yield global_step, accumulated_loss / max(accumulation_step, 1), "⏹️ Training stopped by user"
                    return

                # Forward pass
                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps

                # Backward pass
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                # Optimizer step
                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    self.fabric.clip_gradients(
                        clip_module,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                    # Log — correct loss: multiply back by G to report true per-sample loss
                    avg_loss = accumulated_loss * self.training_config.gradient_accumulation_steps / accumulation_step
                    self.fabric.log("train/loss", avg_loss, step=global_step)
                    self.fabric.log("train/lr", scheduler.get_last_lr()[0], step=global_step)

                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield global_step, avg_loss, f"Epoch {epoch+1}/{self.training_config.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}"

                    epoch_loss += avg_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            # Flush any remaining accumulated gradients at epoch boundary
            if accumulation_step > 0:
                self.fabric.clip_gradients(
                    clip_module,
                    optimizer,
                    max_norm=self.training_config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accumulated_loss * self.training_config.gradient_accumulation_steps / accumulation_step
                epoch_loss += avg_loss
                num_batches += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            # Clear VRAM cache at end of each epoch to reduce fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)

            self.fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)

            # Activate best-loss tracking once we reach the warmup threshold.
            # Reset everything at that point so we only track post-warmup bests.
            if auto_save_best_after > 0 and (epoch + 1) >= auto_save_best_after and not best_tracking_active:
                best_tracking_active = True
                best_loss = float('inf')
                patience_counter = 0
                recent_losses.clear()
                yield global_step, avg_epoch_loss, f"📊 Best-model tracking activated from epoch {epoch+1}"

            # Update rolling window of recent losses
            recent_losses.append(avg_epoch_loss)
            if len(recent_losses) > loss_window_size:
                recent_losses.pop(0)

            # Smoothed loss = moving average over last 5 epochs
            smoothed_loss = sum(recent_losses) / len(recent_losses)

            # Track best smoothed loss (only meaningful after warmup)
            is_new_best = best_tracking_active and smoothed_loss < best_loss - min_delta
            if is_new_best:
                best_loss = smoothed_loss
                patience_counter = 0
                best_epoch = epoch + 1
            elif best_tracking_active:
                patience_counter += 1

            # Build epoch status message — always show MA5, show best only after warmup
            ma5_str = f", MA5: {smoothed_loss:.4f}" if len(recent_losses) >= 2 else ""
            best_str = f" (best: {best_loss:.4f} @ ep{best_epoch})" if best_tracking_active else ""
            yield global_step, avg_epoch_loss, f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}{ma5_str}{best_str}"

            # Auto-save best model (after warmup period)
            if is_new_best:
                best_path = os.path.join(self.training_config.output_dir, "best")
                self._save_adapter_weights(best_path)
                yield global_step, avg_epoch_loss, f"⭐ Best model saved (epoch {epoch+1}, MA5: {best_loss:.4f})"

            # Early stopping (only if enabled, and only after warmup period)
            if early_stop_patience > 0 and best_tracking_active and patience_counter >= early_stop_patience:
                yield global_step, avg_epoch_loss, f"🛑 Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs, best MA5={best_loss:.4f} at epoch {best_epoch})"
                break

            # Save checkpoint
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}")
                self._save_adapter_checkpoint(optimizer, scheduler, epoch + 1, global_step, checkpoint_dir)
                yield global_step, avg_epoch_loss, f"💾 Checkpoint saved at epoch {epoch+1}"

        # Save final model
        # If we have a best model from MA5 tracking, copy it as final.
        # Otherwise save the current (last epoch) weights as final.
        final_path = os.path.join(self.training_config.output_dir, "final")
        best_path = os.path.join(self.training_config.output_dir, "best")
        adapter_label = "LoKr" if self.module.adapter_type == "lokr" else "LoRA"
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        if best_tracking_active and best_epoch > 0 and os.path.exists(best_path):
            # Copy best MA5 model as final output
            import shutil
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(best_path, final_path)
            yield global_step, final_loss, f"✅ Training complete! {adapter_label} final = best MA5 (epoch {best_epoch}, MA5: {best_loss:.4f}) saved to {final_path}"
        else:
            self._save_adapter_weights(final_path)
            yield global_step, final_loss, f"✅ Training complete! {adapter_label} saved to {final_path} (last epoch weights)"

    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Basic training loop without Fabric."""
        # Resolve precision
        resolved_precision = self.training_config.resolve_precision()
        train_dtype = precision_to_dtype(resolved_precision)
        self.module.dtype = train_dtype
        self.module.model = self.module.model.to(train_dtype)

        # For LoKr: ensure lycoris_net params are on correct device/dtype
        # LyCORIS creates adapter params on CPU; they must be co-located with the decoder
        if self.module.adapter_type == "lokr" and self.module.lycoris_net is not None:
            target_device = self.module.device
            self.module.lycoris_net = self.module.lycoris_net.to(device=target_device, dtype=train_dtype)
            self.module.model.decoder = self.module.model.decoder.to(device=target_device)

        # Use GradScaler for fp16 (bf16 and fp32 don't need it)
        use_scaler = resolved_precision == "fp16" and self.module.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None

        yield 0, 0.0, f"🚀 Starting basic training loop (precision: {resolved_precision})..."

        os.makedirs(self.training_config.output_dir, exist_ok=True)

        train_loader = data_module.train_dataloader()

        # Get trainable params from the right source
        if self.module.adapter_type == "lokr" and self.module.lycoris_net is not None:
            trainable_params = list(self.module.lycoris_net.parameters())
        else:
            trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        optimizer = create_optimizer(trainable_params, self.training_config)

        total_steps = len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps

        scheduler = create_scheduler(optimizer, self.training_config, total_steps)

        yield 0, 0.0, f"📋 Optimizer: {self.training_config.optimizer_type}, Scheduler: {self.training_config.scheduler_type}"

        # Encoder offloading: move non-decoder components to CPU
        if self.training_config.encoder_offloading:
            try:
                offloaded = []
                for attr_name in ('encoder', 'text_encoder', 'vae', 'tokenizer'):
                    component = getattr(self.dit_handler.model, attr_name, None)
                    if component is not None and hasattr(component, 'to'):
                        try:
                            component.to('cpu')
                            offloaded.append(attr_name)
                        except Exception:
                            pass
                if offloaded:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    yield 0, 0.0, f"📤 Offloaded to CPU: {', '.join(offloaded)}"
            except Exception as e:
                logger.warning(f"Encoder offloading failed: {e}")

        # Gradient checkpointing
        if self.training_config.gradient_checkpointing:
            try:
                decoder = self.module.model.decoder
                if hasattr(decoder, 'base_model') and hasattr(decoder.base_model, 'model'):
                    raw_decoder = decoder.base_model.model
                else:
                    raw_decoder = decoder
                if hasattr(raw_decoder, 'gradient_checkpointing_enable'):
                    raw_decoder.gradient_checkpointing_enable()
                    yield 0, 0.0, "♻️ Gradient checkpointing enabled"
                else:
                    enabled_count = 0
                    for module in raw_decoder.modules():
                        if hasattr(module, 'gradient_checkpointing_enable'):
                            module.gradient_checkpointing_enable()
                            enabled_count += 1
                    if enabled_count > 0:
                        yield 0, 0.0, f"♻️ Gradient checkpointing enabled on {enabled_count} blocks"
            except Exception as e:
                logger.warning(f"Gradient checkpointing failed: {e}")

        start_epoch = 0
        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0

        # Handle resume from checkpoint
        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."

                checkpoint_info = load_training_checkpoint(
                    resume_from,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                )

                if checkpoint_info["adapter_path"]:
                    adapter_path = checkpoint_info["adapter_path"]
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                    if not os.path.exists(adapter_weights_path):
                        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")

                    if os.path.exists(adapter_weights_path):
                        from safetensors.torch import load_file
                        if adapter_weights_path.endswith(".safetensors"):
                            state_dict = load_file(adapter_weights_path)
                        else:
                            state_dict = torch.load(adapter_weights_path, map_location=self.module.device)

                        self.module.model.decoder.load_state_dict(state_dict, strict=False)

                        start_epoch = checkpoint_info["epoch"]
                        global_step = checkpoint_info["global_step"]

                        status_parts = [f"✅ Resumed from epoch {start_epoch}, step {global_step}"]
                        if checkpoint_info["loaded_optimizer"]:
                            status_parts.append("optimizer restored")
                        if checkpoint_info["loaded_scheduler"]:
                            status_parts.append("scheduler restored")
                        yield 0, 0.0, ", ".join(status_parts)
                    else:
                        yield 0, 0.0, f"⚠️ Adapter weights not found in {adapter_path}"
                else:
                    yield 0, 0.0, f"⚠️ No valid checkpoint found in {resume_from}"

            except Exception as e:
                logger.exception("Failed to load checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
                start_epoch = 0
                global_step = 0
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        # Best loss tracking — only counts after auto_save_best_after warmup.
        # Uses a moving average (window=5) instead of single-epoch loss to
        # avoid saving on random low-loss spikes.
        best_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_tracking_active = False  # flips True once we pass the warmup
        early_stop_patience = getattr(self.training_config, 'early_stop_patience', 0)
        auto_save_best_after = getattr(self.training_config, 'auto_save_best_after', 200)
        min_delta = 0.001
        loss_window_size = 5
        recent_losses = []  # rolling window for smoothed best-loss tracking

        self.module.model.decoder.train()

        for epoch in range(start_epoch, self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    yield global_step, accumulated_loss / max(accumulation_step, 1), "⏹️ Training stopped"
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.training_config.max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Correct loss: multiply back by G to report true per-sample loss
                    avg_loss = accumulated_loss * self.training_config.gradient_accumulation_steps / accumulation_step

                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield global_step, avg_loss, f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}"

                    epoch_loss += avg_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            # Flush remaining accumulated gradients at epoch boundary
            if accumulation_step > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, self.training_config.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accumulated_loss * self.training_config.gradient_accumulation_steps / accumulation_step
                epoch_loss += avg_loss
                num_batches += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            # Clear VRAM cache at end of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)

            # Activate best-loss tracking once we reach the warmup threshold.
            # Reset everything at that point so we only track post-warmup bests.
            if auto_save_best_after > 0 and (epoch + 1) >= auto_save_best_after and not best_tracking_active:
                best_tracking_active = True
                best_loss = float('inf')
                patience_counter = 0
                recent_losses.clear()
                yield global_step, avg_epoch_loss, f"📊 Best-model tracking activated from epoch {epoch+1}"

            # Update rolling window of recent losses
            recent_losses.append(avg_epoch_loss)
            if len(recent_losses) > loss_window_size:
                recent_losses.pop(0)

            # Smoothed loss = moving average over last 5 epochs
            smoothed_loss = sum(recent_losses) / len(recent_losses)

            # Track best smoothed loss (only meaningful after warmup)
            is_new_best = best_tracking_active and smoothed_loss < best_loss - min_delta
            if is_new_best:
                best_loss = smoothed_loss
                patience_counter = 0
                best_epoch = epoch + 1
            elif best_tracking_active:
                patience_counter += 1

            # Build epoch status message — always show MA5, show best only after warmup
            ma5_str = f", MA5: {smoothed_loss:.4f}" if len(recent_losses) >= 2 else ""
            best_str = f" (best: {best_loss:.4f} @ ep{best_epoch})" if best_tracking_active else ""
            yield global_step, avg_epoch_loss, f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}{ma5_str}{best_str}"

            # Auto-save best model (after warmup period)
            if is_new_best:
                best_path = os.path.join(self.training_config.output_dir, "best")
                self._save_adapter_weights(best_path)
                yield global_step, avg_epoch_loss, f"⭐ Best model saved (epoch {epoch+1}, MA5: {best_loss:.4f})"

            # Early stopping (only if enabled, and only after warmup period)
            if early_stop_patience > 0 and best_tracking_active and patience_counter >= early_stop_patience:
                yield global_step, avg_epoch_loss, f"🛑 Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs, best MA5={best_loss:.4f} at epoch {best_epoch})"
                break

            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}")
                self._save_adapter_checkpoint(optimizer, scheduler, epoch + 1, global_step, checkpoint_dir)
                yield global_step, avg_epoch_loss, f"💾 Checkpoint saved"

        # Save final model — prefer best MA5 model over last epoch weights
        final_path = os.path.join(self.training_config.output_dir, "final")
        best_path = os.path.join(self.training_config.output_dir, "best")
        adapter_label = "LoKr" if self.module.adapter_type == "lokr" else "LoRA"
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        if best_tracking_active and best_epoch > 0 and os.path.exists(best_path):
            import shutil
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(best_path, final_path)
            yield global_step, final_loss, f"✅ Training complete! {adapter_label} final = best MA5 (epoch {best_epoch}, MA5: {best_loss:.4f}) saved to {final_path}"
        else:
            self._save_adapter_weights(final_path)
            yield global_step, final_loss, f"✅ Training complete! {adapter_label} saved to {final_path} (last epoch weights)"

    def stop(self):
        """Stop training."""
        self.is_training = False
