# Changelog

## 2026-02-14b — LoKr Support, Upstream Optimizations & Bug Fixes

### LoKr (Low-Rank Kronecker) Training Support

Added full LoKr adapter training as an alternative to LoRA, powered by the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) library.

**New files:**
- `acestep/training/lokr_utils.py` — LoKr injection, save/load/merge utilities
- `acestep/training/configs.py` — New `LoKRConfig` dataclass

**What changed:**
- **Training UI** — New "Adapter Type" radio selector (LoRA / LoKr) with conditional parameter panels. LoKr settings: Factor (-1 = auto), Linear Dim (10000 = auto), Linear Alpha, Decompose Both, Use Tucker, Dropout.
- **Trainer** — `PreprocessedLoRAModule` and `LoRATrainer` now branch on `adapter_type` for injection, save, load, merge, gradient clipping, and Fabric setup. The training loop (flow matching, timestep sampling, CFG dropout) is identical for both adapters.
- **Merge tab** — Auto-detects adapter type from checkpoint files (`adapter_config.json` = PEFT LoRA, `lokr_config.json` = LyCORIS LoKr).
- **Inference loader** (`handler.py`) — `load_lora()` auto-detects and loads both LoRA (PEFT) and LoKr (LyCORIS) adapters. `unload_lora()` cleans up both adapter types.

**LoKr vs LoRA:**
- LoKr uses Kronecker product factorization — often more parameter-efficient at similar expressiveness
- LoKr is experimental and requires `lycoris-lora>=2.0.0` (optional dependency, commented in `requirements.txt`)
- LoRA remains the default and recommended option

### Upstream ACE-Step 1.5 Optimizations Integrated

Integrated key improvements from the upstream [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) repository:

**PR #499 — LoRA Memory Bloat Fix (70% VRAM reduction on load/unload)**
- Replaced `copy.deepcopy(model.decoder)` with CPU `state_dict()` backup for the base decoder
- Loading a LoRA/LoKr adapter no longer doubles VRAM usage (~10-15GB saved)
- Unloading restores from CPU state_dict + calls `torch.cuda.empty_cache()`
- VRAM usage is logged before/after load and unload operations

**PR #554 — Opus and AAC Output Formats**
- `AudioSaver` now supports `opus` and `aac` formats via ffmpeg backend
- Added to format validation, extension handling, and convert functions
- Gradio UI dropdown updated with new format options

**PR #492 — Batch LLM Load/Offload Optimization**
- `_load_model_context()` made reentrant: if model is already on GPU, nested calls are no-ops
- Batch loop in `_run_pt()` wrapped in a single `_load_model_context()` so the LLM loads once and offloads once per batch (was N loads + N offloads before)

**PR #493 — Phase-Aware max_new_tokens / Progress Bar Fix**
- New `_compute_max_new_tokens()` method replaces inline calculations
- In "codes" phase, buffer is +10 tokens (not +500) since constrained decoder forces exact EOS
- Fixes progress bar appearing to stall at ~66% during codes generation

### Bug Fix: Checkpoint Discovery for Pinokio Layout

- `get_checkpoints_dir()` now searches both `ACE-Step-1.5/checkpoints/` and `checkpoints/` directories
- New `_get_all_checkpoint_dirs()` scans all known checkpoint locations with deduplication
- New `resolve_checkpoint_path()` resolves checkpoint names to full paths across all directories
- Fixes empty "Model Checkpoint" dropdown when running under Pinokio (where base models are symlinked to `ACE-Step-1.5/checkpoints/`, not `checkpoints/`)

---

## 2026-02-14 — Training Quality Fix: Corrected Timestep Sampling & CFG Dropout

### Background — How We Found These Issues

While reviewing the upstream [ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5), we noticed that the **Side-Step training v2** module ([PR #478](https://github.com/ace-step/ACE-Step-1.5/pull/478) by [@koda-dernet](https://github.com/koda-dernet)) was merged on Feb 13. This module introduced a completely new `acestep/training_v2/` directory with corrected training algorithms.

After a thorough analysis comparing our `acestep/training/trainer.py` (v1) against the new `trainer_fixed.py` (v2), we identified **4 critical correctness issues** in our training pipeline that were degrading LoRA quality. These issues existed in the original training code that we inherited, not in our own additions.

The key insight from Side-Step's code is that the original v1 trainer used timestep sampling and CFG dropout strategies designed for **inference**, not for **LoRA training**. The v2 trainer corrects these to match the model's actual **pre-training** distribution.

### Critical Fix 1: Timestep Sampling — Discrete → Logit-Normal

**Problem:** Our trainer (v1) used a fixed set of 8 discrete timesteps for turbo models, sampled uniformly from `[1.0, 0.95, 0.9, 0.83, 0.75, 0.64, 0.5, 0.3]`. This schedule was designed for the turbo model's 8-step **inference** pipeline, not for training.

**Why it matters:** During pre-training, ACE-Step uses a **logit-normal distribution** (`t = sigmoid(N(μ, σ))` with `μ=-0.4, σ=1.0`) to sample timesteps. By training LoRA with only 8 discrete timesteps, we were teaching the model to only denoise at those specific noise levels, creating a distribution mismatch with the full continuous spectrum the model actually uses.

**Fix:** Replaced `sample_discrete_timestep()` and `sample_continuous_timestep()` with a single `sample_logit_normal_timestep()` that faithfully reproduces the model's own `sample_t_r()` function (from `modeling_acestep_v15_turbo.py` lines 169-194). This is used for **all** model variants (turbo and base).

**Source:** `acestep/training_v2/timestep_sampling.py:sample_timesteps()` in [ACE-Step 1.5 PR #478](https://github.com/ace-step/ACE-Step-1.5/pull/478)

### Critical Fix 2: CFG Dropout — Zeroed Embeddings → Learned Null Embedding

**Problem:** When applying classifier-free guidance (CFG) dropout during training, our v1 code **zeroed out** the encoder hidden states for dropped samples. Additionally, CFG dropout was only applied to base models, not turbo.

**Why it matters:** The model was pre-trained with a specific **learned null/unconditional embedding** (`model.null_condition_emb`), not zero vectors. Zeroing out embeddings creates an artificial input the model never saw during pre-training, producing incorrect gradient signals. Furthermore, the original pre-training applied CFG dropout to all model variants, not just base.

**Fix:** New `apply_cfg_dropout()` function that:
1. Uses `model.null_condition_emb` (the model's learned unconditional embedding) instead of zeros
2. Is applied to **all model types** (turbo + base), matching pre-training behavior
3. Default dropout probability changed from 0.1 to 0.15 (matching v2 defaults)

**Source:** `acestep/training_v2/timestep_sampling.py:apply_cfg_dropout()` and `trainer_fixed.py` lines referencing `model.null_condition_emb` in [ACE-Step 1.5 PR #478](https://github.com/ace-step/ACE-Step-1.5/pull/478)

### Critical Fix 3: Loss Reporting — Corrected Arithmetic

**Problem:** When gradient accumulation > 1, our reported loss values were divided by `gradient_accumulation_steps` (because each micro-batch loss is already divided by G before backward). The logged/displayed loss was `1/G` of the actual per-sample loss.

**Fix:** Corrected loss computation: `avg_loss = accumulated_loss * gradient_accumulation_steps / accumulation_step` to report the true per-sample loss value.

**Note:** If you compare loss curves from before and after this fix, post-fix values will appear higher by a factor of G. The underlying training is identical — only the reported number changed.

### Critical Fix 4: End-of-Epoch Remainder Flushing

**Problem:** When the number of batches per epoch is not evenly divisible by `gradient_accumulation_steps`, the leftover accumulated gradients at the end of each epoch were silently dropped. For example, with 17 songs and gradient_accumulation=4, the last 1 sample's gradients were thrown away every epoch.

**Fix:** Added explicit remainder flushing at epoch boundaries — if there are accumulated gradients when the epoch ends, an optimizer step is performed to use them.

### New UI Parameters

- **Timestep μ (mu)** — Logit-normal mean. Default -0.4 (bias toward cleaner data, matching pre-training). Available in Advanced accordion.
- **Timestep σ (sigma)** — Logit-normal standard deviation. Default 1.0. Available in Advanced accordion.
- **CFG Dropout** — Moved from base-model-only to a general parameter. Default 0.15 (15%). Now applies to all model types.

### Other Improvements
- **Per-epoch VRAM cache clearing** — Calls `torch.cuda.empty_cache()` at end of each epoch to reduce VRAM fragmentation
- Training start message now shows timestep distribution parameters: `logit-normal(μ=-0.4, σ=1.0)`

### What We Kept (Our Features Not in v2)
The v2 Side-Step trainer removes several features that we intentionally keep because they improve the training experience:
- ✅ Early stopping with MA5 smoothed loss tracking
- ✅ Auto-save best model after warmup period
- ✅ torch.compile support
- ✅ VRAM monitor and pre-training GPU check
- ✅ GPU presets for different hardware tiers

---

## 2026-02-13 — Initial Release

First public release of the ACE-Step LoRA Trainer + Captioner as a standalone project, separated from the Pinokio launcher.

### Repository Setup
- **Initial commit** — Extracted core training modules (`acestep/training/`), model handler, LLM inference, captioner, and Gradio UIs from the Pinokio-based ACE-Step project into a self-contained standalone repository
- 20,000+ lines of code across 26 files
- Published to GitHub at `github.com/Estylon/ace-lora-trainer`

### Bug Fixes
- **Fixed ACESTEP_PATH pointing to non-existent directory** — Was looking for `ACE-Step-1.5/` which doesn't exist in standalone repo. Changed to repo root so checkpoint discovery works correctly
- **Fixed captioner/transcriber hardcoded Pinokio paths** — Changed `DEFAULT_MODEL_PATH` from `D:/pinokio/drive/.../acestep-captioner` to HuggingFace model ID `ACE-Step/acestep-captioner` so models auto-download on first use
- **Fixed captioner rejecting HuggingFace model IDs** — `os.path.exists()` was blocking valid HF model IDs like `ACE-Step/acestep-captioner`. Added HF ID detection (org/model format) to allow them through to `from_pretrained()`
- **Fixed torchao breaking Windows installs** — torchao has no Windows wheels and requires `torch.compile` + Triton (neither available on Windows). Made it Linux-only in `requirements.txt` since it's only needed for INT8/FP8 quantization, not LoRA training
- **Fixed RTX 3060/4060 GPU preset** — Changed optimizer from `adamw8bit` to `adafactor` and scheduler from `cosine` to `constant_with_warmup`. Adafactor uses near-zero optimizer state memory, making it viable for 8GB VRAM where AdamW 8-bit still OOMs

### Dependency Fixes
- Added missing `pydub` (audio splitting) and `diskcache` (caching) to requirements
- Removed unused `xxhash` dependency
- Added missing `acestep/genres_vocab.txt` (178K genre vocabulary file needed for caption generation)
- Updated `.gitignore` with `.cache/`, `lora_projects/`, `datasets/*.json`

### New Features

#### Install & Launch Scripts
- **`install.bat`** (Windows) and **`install.sh`** (Linux/Mac) — One-click installer that creates virtual environment, installs all dependencies via `uv pip` (with pip fallback), and verifies critical packages (PyTorch, PEFT, Lightning, Gradio, Prodigy, Transformers)
- **`start.bat`** (Windows) and **`start.sh`** (Linux/Mac) — One-click launcher that activates venv and runs the trainer
- **Environment checks in `launch.py`** — Detects if running without a virtual environment or with missing packages (especially PEFT) and warns before proceeding. This prevents the silent failure where training falls back to full fine-tuning without LoRA (~35x larger checkpoints, much slower)

#### VRAM Monitor
- **Real-time GPU status** in the Training tab showing GPU name, VRAM used/free/total with color-coded indicator (green/yellow/red)
- **Other process detection** — Uses `nvidia-smi` to list other processes occupying the GPU (e.g., ACE-Step API) with their VRAM usage
- **Pre-training VRAM check** — When clicking "Start Training", automatically checks for insufficient free VRAM and warns about other GPU processes
- **Refresh button** for manual VRAM status updates

### Documentation
- **`TRAINING_GUIDE.md`** — Comprehensive ~950-line training guide covering complete workflow (7 steps from captioning to merge), all parameters with defaults/ranges/explanations, all 4 GPU presets with full parameter tables, optimizer/scheduler comparison tables, disk space requirements, security audit section, and troubleshooting guide
- **`LICENSE`** — Apache License 2.0
- **`README.md`** — Updated with accurate GPU preset table (including scheduler column), auto-download model documentation, simplified install instructions using new scripts, and links to training guide

### Architecture Notes
The standalone trainer is **100% self-sufficient** — all model loading and inference happens in-process via PyTorch/HuggingFace Transformers. No external API server is required. The Gradio UI handles everything: model initialization, dataset building, auto-labeling, preprocessing, training, and LoRA merging.
