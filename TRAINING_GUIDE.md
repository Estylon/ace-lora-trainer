# ACE-Step LoRA Training Guide

A complete, in-depth guide to training LoRA adapters on ACE-Step 1.5 music generation models using this trainer.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Disk Space Requirements](#disk-space-requirements)
5. [The Two Interfaces](#the-two-interfaces)
6. [Complete Training Workflow](#complete-training-workflow)
   - [Step 1: Captioning Your Audio](#step-1-captioning-your-audio)
   - [Step 2: Load the Model](#step-2-load-the-model)
   - [Step 3: Build Your Dataset](#step-3-build-your-dataset)
   - [Step 4: Preprocess Tensors](#step-4-preprocess-tensors)
   - [Step 5: Configure Training](#step-5-configure-training)
   - [Step 6: Train](#step-6-train)
   - [Step 7: Export or Merge](#step-7-export-or-merge)
7. [Understanding the Parameters](#understanding-the-parameters)
   - [LoRA Architecture](#lora-architecture)
   - [Optimizers](#optimizers)
   - [Schedulers](#schedulers)
   - [Attention Targeting](#attention-targeting)
   - [VRAM Saving Features](#vram-saving-features)
   - [Loss, Early Stopping & Best Model](#loss-early-stopping--best-model)
   - [Random Crop Augmentation](#random-crop-augmentation)
   - [Turbo vs Base Model Training](#turbo-vs-base-model-training)
8. [GPU Presets Reference](#gpu-presets-reference)
9. [Epoch & Dataset Guidelines](#epoch--dataset-guidelines)
10. [Advanced Features](#advanced-features)
    - [Gradient Sensitivity Estimation](#gradient-sensitivity-estimation)
    - [Resume from Checkpoint](#resume-from-checkpoint)
    - [LoRA Merge into Base Model](#lora-merge-into-base-model)
11. [Captioner Deep Dive](#captioner-deep-dive)
12. [Troubleshooting](#troubleshooting)
13. [Security Notes](#security-notes)

---

## Overview

This tool lets you fine-tune ACE-Step 1.5 music generation models using **LoRA** (Low-Rank Adaptation) or **LoKr** (Low-Rank Kronecker, via LyCORIS). Instead of retraining the entire 4.5GB model, these adapters train small weights (43-85MB) that modify the model's behavior. This means:

- **Fast training** — minutes to hours instead of days
- **Small output** — adapter files are 43-85MB, not gigabytes
- **Stackable** — multiple adapters can be swapped without retraining
- **Reversible** — the base model is never modified
- **Two adapter types** — LoRA (default, proven) or LoKr (experimental, potentially more parameter-efficient)

The trainer works with both **turbo** (8-step, fast inference) and **base** (60-step, higher quality) ACE-Step variants.

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM (RTX 3060) | 24GB VRAM (RTX 4090) |
| RAM | 16GB | 32GB |
| Storage | 30GB free | 60GB+ free |
| OS | Windows 10/11, Linux, macOS | Windows 11, Ubuntu 22.04+ |

### Software

- Python 3.10 or 3.11
- CUDA 12.1+ (for NVIDIA GPUs)
- Git

---

## Installation

```bash
git clone https://github.com/Estylon/ace-lora-trainer.git
cd ace-lora-trainer

# Create virtual environment
python -m venv env
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# Install dependencies (uv is faster, pip works too)
uv pip install -r requirements.txt
# Or: pip install -r requirements.txt
```

### Optional: 8-bit Optimizer Support

If you want to use AdamW 8-bit (saves ~30-40% optimizer VRAM), install bitsandbytes separately:

```bash
# Windows/Linux only (not supported on macOS)
uv pip install bitsandbytes>=0.43.0
```

### Optional: LoKr (LyCORIS) Support

If you want to train LoKr adapters (alternative to LoRA), install lycoris-lora:

```bash
uv pip install lycoris-lora>=2.0.0
```

### Models (Auto-Download)

All models auto-download from HuggingFace on first use. No manual download needed.

| Model | Size | What It Does |
|-------|------|--------------|
| ACE-Step v1.5 Turbo | ~6GB total | The generation model you'll fine-tune |
| ACE-Step Captioner | ~22GB | Generates music descriptions from audio |
| ACE-Step Transcriber | ~22GB | Extracts structured lyrics from audio |

The generation model includes several components that download together: the DiT decoder (~4.5GB), VAE (~322MB), text encoder Qwen3-Embedding-0.6B (~1.2GB), and language model acestep-5Hz-lm-1.7B (~45MB).

---

## Disk Space Requirements

Training LoRAs generates checkpoints that accumulate on disk. Here's what to expect with real-world numbers measured from actual training runs.

### Per-Song Preprocessed Tensors

Each song gets converted to a `.pt` tensor file during preprocessing:

| Song Duration | Tensor Size |
|---------------|-------------|
| 3 minutes | ~4.5 MB |
| 4 minutes | ~4.8 MB |
| 5 minutes | ~5.1 MB |

**Rule of thumb: ~5 MB per song.**

### Checkpoint Sizes

Each saved checkpoint contains the LoRA adapter weights plus the optimizer/scheduler state for resuming training. The "best" and "final" saves contain only the adapter (no optimizer state).

| LoRA Rank | Adapter Only (best/final) | Full Checkpoint (adapter + optimizer state) |
|-----------|---------------------------|---------------------------------------------|
| 16 | ~11 MB | ~32 MB |
| 32 | ~43 MB | ~128 MB |
| 64 | ~85 MB | ~253 MB |

### Total Disk Space Estimates

Here's how disk usage accumulates during a typical training run, including the `best/` and `final/` saves:

| Scenario | Epochs | Save Every | Rank | Checkpoints Created | Total Disk |
|----------|--------|------------|------|---------------------|------------|
| **Small dataset, 4090** | 800 | 50 | 64 | 16 regular + best + final | **~4.2 GB** |
| **Small dataset, 3090** | 800 | 50 | 64 | 16 regular + best + final | **~4.2 GB** |
| **Small dataset, 3080** | 1000 | 50 | 32 | 20 regular + best + final | **~2.6 GB** |
| **Small dataset, 3060** | 1000 | 50 | 16 | 20 regular + best + final | **~0.7 GB** |
| **Large dataset, 4090** | 300 | 50 | 64 | 6 regular + best + final | **~1.7 GB** |

### Merged Model Size

If you merge a LoRA into the base model, the output is a full standalone model:

| Output | Size |
|--------|------|
| Merged model (safetensors) | ~4.5 GB |
| silence_latent.pt | ~3.7 MB |

### Summary: Total Project Disk Usage

| What | Space |
|------|-------|
| This repo (code) | ~2 MB |
| Python venv + dependencies | ~8-12 GB |
| ACE-Step model (auto-downloaded) | ~6 GB |
| Captioner model (if used) | ~22 GB |
| Transcriber model (if used) | ~22 GB |
| Training tensors (10 songs) | ~50 MB |
| Training checkpoints (typical run) | ~1-4 GB |
| **Total (trainer only, no captioner)** | **~16-22 GB** |
| **Total (with captioner + transcriber)** | **~58-66 GB** |

> **Tip:** You can delete old checkpoints from `lora_output/<project>/checkpoints/` once you've identified the best epoch. Keep only `best/` and `final/`.

---

## The Two Interfaces

This tool ships two **separate** Gradio web UIs:

### Captioner UI — `python launch.py --mode caption`
Analyzes your audio files and generates:
- **Captions** — Detailed music style descriptions (genre, instruments, structure, mood)
- **Metadata** — BPM, musical key, time signature, duration, genre, language
- **Lyrics** — Structured lyrics with `[Verse]`, `[Chorus]`, `[Bridge]` section tags

These outputs become the training labels for your LoRA.

### Trainer UI — `python launch.py --mode train`
The full training pipeline:
- Load an ACE-Step checkpoint
- Build and edit your dataset
- Preprocess audio into training tensors
- Configure and run training
- Export the trained LoRA adapter

### Launch Modes

```bash
python launch.py                  # Training UI only (port 7861)
python launch.py --mode caption   # Captioner UI only (port 7861)
python launch.py --mode both      # Both UIs (trainer on 7861, captioner on 7862)
```

---

## Complete Training Workflow

### Step 1: Captioning Your Audio

**Goal:** Generate text descriptions and metadata for each audio file, so the model knows what it's learning.

1. Launch the captioner: `python launch.py --mode caption`
2. Open http://127.0.0.1:7861 in your browser
3. Enter the path to your audio folder (supports `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`)
4. Click **"Load Captioner Model"** — the model (`ACE-Step/acestep-captioner`, ~22GB) downloads automatically on first use
5. Click **"Caption All"** — processes each file, generating:
   - A detailed text caption describing the music
   - BPM via librosa beat tracking
   - Musical key, time signature, and genre via AI analysis
6. Optionally load the transcriber to extract structured lyrics
7. Each audio file gets a `.json` sidecar with all metadata

**VRAM-Saving Two-Pass Workflow (recommended for <24GB VRAM):**
1. Load Captioner → Caption All → **Unload Captioner** (frees ~22GB VRAM)
2. Load Transcriber → Transcribe Lyrics Only → **Unload Transcriber**

**Output format** — each audio file gets a JSON sidecar:
```json
{
  "filename": "song_01.mp3",
  "caption": "A high-energy rock track with distorted electric guitar riffs...",
  "bpm": 142,
  "keyscale": "E Minor",
  "timesignature": "4",
  "genre": "rock",
  "lyrics": "[Verse 1]\nWalking down the empty street...\n[Chorus]\nWe are the ones...",
  "language": "en",
  "is_instrumental": false,
  "duration": 234
}
```

### Step 2: Load the Model

1. Launch the trainer: `python launch.py --mode train`
2. Open http://127.0.0.1:7861
3. In the **"1. Load Model"** section, select a checkpoint from the dropdown (e.g., `acestep-v15-turbo`)
4. Click **"Initialize Service"**

The trainer auto-detects whether you selected a **turbo** or **base** model and configures parameters accordingly. Both model types use logit-normal timestep sampling and CFG dropout with the model's learned null embedding — the parameters are read from `config.json`.

### Step 3: Build Your Dataset

1. In the **"2. Build Dataset"** section, enter the path to your audio folder
2. Set a **Dataset Name** (e.g., `my_artist_lora`)
3. Set an **Activation Tag** — a unique trigger word (e.g., `ZX_MyArtist`)
   - This tag gets injected into every caption so the model associates it with your style
   - Use a distinctive prefix like `ZX_` to avoid conflicts with real words
4. Choose **Tag Position**:
   - `replace` (default) — replaces the caption entirely with just the tag
   - `prepend` — adds the tag before the caption
   - `append` — adds the tag after the caption
5. Click **"Scan Audio Directory"** — finds all audio files and loads any existing caption JSONs
6. Optionally click **"Auto-Label"** to use the loaded LLM for captioning (alternative to the standalone captioner)
7. **Review and edit** individual samples using the sample editor — you can adjust captions, genres, lyrics, BPM, key, etc.
8. Click **"Save Dataset"** — saves a JSON manifest with all your labels

### Step 4: Preprocess Tensors

1. In the **"3. Preprocess"** section, verify the tensor output directory
2. Set **Max Duration** (default 240s) — longer songs get truncated
3. Click **"Preprocess Tensors"**

This step encodes each audio file through the VAE and text encoder, producing `.pt` tensor files (~5MB each). These tensors are what the training loop actually trains on.

### Step 5: Configure Training

You can either select a **GPU Preset** for one-click configuration, or manually tune every parameter.

**Quick setup with GPU Presets:**
1. Select your GPU tier from the dropdown (e.g., "RTX 4090 / 5090")
2. All 18 parameters are auto-configured optimally for your VRAM

**Or configure manually** — see [Understanding the Parameters](#understanding-the-parameters) below for details on every setting.

### Step 6: Train

1. Click **"Start Training"**
2. Watch the live loss plot — loss should decrease over time
3. The log window shows epoch progress, loss values, learning rate, and checkpoints saved
4. Training ends when:
   - Max epochs reached, or
   - Early stopping triggers (if enabled), or
   - You click "Stop Training"

**What happens during training:**
- Each epoch iterates over your entire dataset
- Audio tensors are randomly cropped to 60s windows each epoch (data augmentation)
- The model learns to predict the noise flow field, minimizing MSE loss
- Checkpoints are saved at regular intervals to `lora_output/<project>/checkpoints/`
- The best model (lowest smoothed loss) is saved to `lora_output/<project>/best/`
- Final weights are always saved to `lora_output/<project>/final/`

### Step 7: Export or Merge

After training, you have two options:

**Option A: Use the adapter directly**
- The adapter files in `best/adapter/` or `final/adapter/` can be loaded at inference time
- Small file size (~43-85MB depending on rank)
- Requires the base model + adapter at inference
- Works for both LoRA and LoKr adapters

**Option B: Merge adapter into base model**
- Creates a standalone model with the adapter baked in
- No need to load the adapter separately at inference
- Output is a full safetensors file (~4.5GB)
- Go to the **"Merge"** section, select base model + adapter checkpoint, click **"Merge"**
- The merge tab auto-detects adapter type (LoRA or LoKr) from checkpoint files

---

## Understanding the Parameters

### Adapter Type: LoRA vs LoKr

The trainer supports two adapter types, selectable via the **Adapter Type** radio in the UI:

| | LoRA (default) | LoKr (experimental) |
|---|---|---|
| **Library** | PEFT (HuggingFace) | LyCORIS |
| **Method** | Low-Rank decomposition (A × B) | Kronecker product factorization |
| **Maturity** | Proven, widely used | Experimental, fewer users |
| **Install** | `peft` (included) | `lycoris-lora>=2.0.0` (optional) |
| **File format** | `adapter_config.json` + `adapter_model.safetensors` | `lokr_config.json` + `lokr_weights.safetensors` |

Both adapters target the same attention layers and use the same training loop (flow matching, timestep sampling, CFG dropout). The only difference is how the weight modifications are factorized.

**When to use LoKr:** If you want to experiment with potentially more parameter-efficient factorization. LoKr can sometimes achieve similar quality with fewer trainable parameters. However, LoRA is better tested and recommended for most users.

### LoRA Parameters

| Parameter | Default | Range | What It Does |
|-----------|---------|-------|--------------|
| **Rank (r)** | 64 | 4-256 | Capacity of the adapter. Higher = more expressive but more VRAM and disk. 16 for subtle style, 32 for solid training, 64 for maximum fidelity. |
| **Alpha** | 128 | 4-512 | Scaling factor. Common practice: set to 2× rank. Higher alpha = stronger LoRA effect. |
| **Dropout** | 0.1 | 0.0-0.5 | Regularization. 0.1 is a safe default. Increase to 0.15-0.2 if overfitting on small datasets. |

### LoKr Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| **Factor** | -1 (auto) | Kronecker factor. -1 = automatic (sqrt of dimension). Controls how the weight matrix is decomposed. |
| **Linear Dim** | 10000 (auto) | Rank for the linear component. 10000 signals auto-selection based on factor. |
| **Linear Alpha** | 1.0 | Scaling factor for the LoKr adapter. |
| **Decompose Both** | Off | If on, applies Kronecker decomposition to both factors (more parameters, more expressive). |
| **Use Tucker** | Off | If on, uses Tucker decomposition for even finer factorization. |
| **Dropout** | 0.0 | Dropout for LoKr layers. |

**How adapters work in this model:**
Both LoRA and LoKr are injected into the DiT (Diffusion Transformer) decoder's attention layers. Specifically, they target the `q_proj`, `k_proj`, `v_proj`, and `o_proj` linear projections inside every attention block. The encoder, VAE, and tokenizer are **never modified** — only the decoder learns.

### Optimizers

| Optimizer | Best For | Memory | How It Works |
|-----------|----------|--------|--------------|
| **AdamW** | General training | High (~2× model params) | Standard optimizer with weight decay. Reliable default. |
| **AdamW 8-bit** | 10-16GB VRAM | Medium (~1.2× model params) | Same as AdamW but stores momentum in 8-bit, saving ~30-40% optimizer VRAM. Requires `bitsandbytes`. |
| **Adafactor** | 8GB VRAM | Very Low (~0.01× model params) | Stores almost no optimizer state. Uses row/column factored second moments. Best option when VRAM is critically limited. |
| **Prodigy** | 16GB+ VRAM, best results | High (~3× model params) | Auto-tunes its own learning rate. Requires `prodigyopt`. Set LR to any value — Prodigy ignores it and finds the optimal rate itself. Paired automatically with a constant scheduler. |

**Prodigy details:** When Prodigy is selected, the trainer forces `LR=1.0` internally (if you set LR ≤ 1e-3) and overrides the scheduler to `constant`. This is by design — Prodigy handles its own learning rate schedule.

**Fallback behavior:** If a required library isn't installed (e.g., `bitsandbytes` for AdamW 8-bit), the trainer automatically falls back to standard AdamW with a warning.

### Schedulers

All schedulers include a **warmup phase** — the learning rate ramps up linearly from 10% to 100% over the first N steps (capped at 10% of total steps).

| Scheduler | After Warmup | Best Paired With |
|-----------|-------------|------------------|
| **Cosine** | LR follows a cosine curve down to 1% of initial | AdamW, AdamW 8-bit |
| **Linear** | LR decreases linearly to 1% of initial | AdamW, AdamW 8-bit |
| **Constant** | LR stays at 100% forever | Prodigy (forced) |
| **Constant + Warmup** | LR ramps up then stays at 100% | Adafactor, Prodigy |

### Attention Targeting

Controls which attention layers receive LoRA adapters:

| Mode | What Gets Trained | When To Use |
|------|-------------------|-------------|
| **Both** (default) | Self-attention + Cross-attention | Maximum expressiveness. Use with 16GB+ VRAM. |
| **Self** | Self-attention only | Saves ~40% adapter parameters. Use on 8GB GPUs or for subtle style transfer. |
| **Cross** | Cross-attention only | Trains how the model responds to text conditioning. Experimental. |

**Technical detail:** Self-attention layers handle how the audio relates to itself (rhythm, structure). Cross-attention layers handle how the audio relates to the text prompt (genre, style, instruments).

### VRAM Saving Features

| Feature | VRAM Saved | Speed Cost | How It Works |
|---------|-----------|------------|--------------|
| **Gradient Checkpointing** | ~40-60% | ~30% slower | Instead of storing all intermediate activations, recomputes them during backward pass. |
| **Encoder Offloading** | ~2-4 GB | Minimal | Moves the text encoder, VAE, and tokenizer to CPU during training (they're not needed after preprocessing). |
| **Attention Type: Self Only** | ~40% adapter | None | Simply trains fewer parameters. |
| **Lower Rank** | Proportional | None | Rank 16 uses ~4× less memory than rank 64. |

**Recommended stacking for low VRAM:**
- 8GB: Gradient Checkpointing + Encoder Offloading + Self-attn only + Rank 16 + Adafactor + Batch 1
- 10-12GB: Gradient Checkpointing + Encoder Offloading + Both attn + Rank 32 + AdamW 8-bit + Batch 1
- 16-24GB: No VRAM saving needed. Go with Prodigy + Rank 64 + Batch 2-3.

### Loss, Early Stopping & Best Model

**Flow Matching Loss:**
The model learns by predicting the "flow" between noise and data. Loss is the MSE (Mean Squared Error) between the predicted flow and the actual flow. Lower loss = better model.

**Best Model Tracking:**
- Tracking only activates after the **Auto-Save Best After** warmup (default: 200 epochs)
- Uses a **Moving Average over 5 epochs (MA5)** to smooth loss fluctuations
- A new best is saved only if `smoothed_loss < best_loss - 0.001` (prevents saving on noise)
- Best model saved to `output_dir/best/adapter/`

**Early Stopping:**
- Also activates only after the warmup period
- Counts how many epochs pass without a new best (patience counter)
- When `patience_counter >= patience_value`, training stops
- Default patience: 80 epochs (meaning 80 epochs without improvement)

**Why the warmup?** The first 100-200 epochs are volatile — loss jumps around as the model starts learning. Tracking best/early-stop too early would give false signals.

### Random Crop Augmentation

**What it does:** If your songs are longer than the configured crop length (default: 1500 frames = 60 seconds), each epoch randomly selects a different 60-second window from each song. This means the model sees different parts of the same song across epochs — natural data augmentation without duplicating files.

**How it works internally:**
- Audio tensors (latents) are cropped along the time axis to `max_latent_length` frames
- Attention masks and context latents are cropped identically
- **Text embeddings are NOT cropped** — they represent the full song description and are not time-aligned
- A new random window is selected every time a sample is loaded (every epoch)

**Frame rate:** 25 frames/second. So 1500 frames = 60s, 3000 frames = 120s.

| Setting | Frames | Duration | When To Use |
|---------|--------|----------|-------------|
| 1500 (default) | 1500 | 60s | Standard training, good variety |
| 1000 | 1000 | 40s | Low VRAM (smaller batches) |
| 3000 | 3000 | 120s | More context per sample, more VRAM |
| 0 | Full song | Full | No cropping — use full songs (high VRAM) |

### Turbo vs Base Model Training

Both model types now use the **same corrected training algorithm** — logit-normal timestep sampling and CFG dropout with the model's learned null embedding. The parameters (`timestep_mu`, `timestep_sigma`) are read from each model's `config.json` automatically.

| Aspect | Turbo | Base / SFT |
|--------|-------|------------|
| **Timestep sampling** | Logit-normal (μ=-0.4, σ=1.0 from config) | Logit-normal (μ=-0.4, σ=1.0 from config) |
| **CFG dropout** | 15% with `null_condition_emb` | 15% with `null_condition_emb` |
| **Inference speed** | Fast (8 steps) | Slower (60 steps) but potentially higher quality |
| **Inference shift** | 3.0 | 1.0 |

**When to use which:**
- **Turbo** — faster iteration, good for style transfer, most common choice
- **Base / SFT** — maximum quality, fully supported with correct training (logit-normal timesteps match pre-training distribution)

---

## GPU Presets Reference

### RTX 4090 / 5090 (24GB+)

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Dropout | 0.1 |
| Learning Rate | 1e-4 |
| Max Epochs | 800 |
| Batch Size | 3 |
| Gradient Accumulation | 1 |
| Optimizer | Prodigy |
| Scheduler | Cosine |
| Attention Type | Both |
| Gradient Checkpointing | Off |
| Encoder Offloading | Off |
| torch.compile | Off |
| Early Stop | On (patience 80) |
| Auto-Save Best After | 200 |
| Save Every N Epochs | 50 |
| Max Crop Length | 1500 (60s) |

### RTX 3090 / 4080 (16-24GB)

Same as 4090 except:
- Batch Size: **2** (instead of 3)
- Gradient Accumulation: **2** (instead of 1)

Effective batch size stays at 4 (2 × 2 accumulation).

### RTX 3080 / 4070 (10-12GB)

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Dropout | 0.1 |
| Learning Rate | 1e-4 |
| Max Epochs | 1000 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Optimizer | AdamW 8-bit |
| Scheduler | Cosine |
| Attention Type | Both |
| Gradient Checkpointing | **On** |
| Encoder Offloading | **On** |
| torch.compile | Off |
| Early Stop | On (patience 80) |
| Auto-Save Best After | 200 |
| Save Every N Epochs | 50 |
| Max Crop Length | 1500 (60s) |

### RTX 3060 / 4060 (8GB)

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Dropout | 0.1 |
| Learning Rate | 1e-4 |
| Max Epochs | 1000 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Optimizer | **Adafactor** |
| Scheduler | **Constant + Warmup** |
| Attention Type | **Self only** |
| Gradient Checkpointing | **On** |
| Encoder Offloading | **On** |
| torch.compile | Off |
| Early Stop | On (patience 80) |
| Auto-Save Best After | 200 |
| Save Every N Epochs | 50 |
| Max Crop Length | **1000 (40s)** |

> **Why Adafactor for 8GB?** AdamW 8-bit still stores significant optimizer state. With only 8GB VRAM minus the model, adapter, activations, and gradients, there's almost no room left for optimizer state. Adafactor uses factored second moments that require near-zero extra memory.

---

## Epoch & Dataset Guidelines

| Number of Songs | Recommended Epochs | Save Every | Expected Training Time (4090) |
|----------------|-------------------|------------|-------------------------------|
| 1-3 | 1500 | 200 | ~30-45 min |
| 4-6 | 1000 | 200 | ~30-50 min |
| 7-10 | 700 | 100 | ~40-60 min |
| 11-20 | 500 | 100 | ~45-90 min |
| 21-50 | 300 | 50 | ~60-120 min |
| 50+ | 200 | 50 | Varies |

**With Early Stopping enabled**, training often converges before max epochs. The early stop patience (default: 80 epochs) means it will stop after 80 consecutive epochs without improvement.

**Tips:**
- More songs = fewer epochs needed (more data diversity per epoch)
- Random crop augmentation effectively multiplies your dataset — a 5-minute song has ~5 possible 60s windows
- The best checkpoint is not always the final one — check the `best/` directory
- If loss plateaus very early, try increasing the learning rate or rank
- If loss oscillates wildly, try decreasing the learning rate or increasing batch size

---

## Advanced Features

### Gradient Sensitivity Estimation

**What it does:** Before training, estimates which attention layers in the model are most sensitive to your specific dataset. This helps you understand which parts of the model matter most for learning your audio style.

**How it works:**
1. Temporarily enables gradients on all attention projection layers
2. Runs N forward+backward passes (default: 10) using your preprocessed tensors
3. Accumulates the L2 norm of gradients for each attention module
4. Ranks modules by gradient magnitude — higher = more sensitive to your data

**Granularity modes:**
- **Layer**: Groups by attention block (e.g., `layers.5.self_attn`) — gives you a per-layer overview, averaging across q/k/v/o projections
- **Module**: Individual projections (e.g., `layers.5.self_attn.q_proj`) — fine-grained view

**How to use it:**
1. Load model and preprocess tensors first
2. Go to the Gradient Estimation accordion
3. Set number of batches (10 is a good default)
4. Click "Run Estimation"
5. Review the ranked table — score is normalized 0.0 to 1.0 relative to the most sensitive module

**Practical use:** If certain layers score near 0, those layers barely react to your data. You could consider targeting only self-attention or only cross-attention based on the results.

### Resume from Checkpoint

Training can be resumed from any saved checkpoint. The checkpoint contains:
- LoRA adapter weights
- Optimizer state (momentum, variance estimates)
- Scheduler state (current step, LR)
- Epoch number

**To resume:**
1. Enable "Resume from Checkpoint" in the training section
2. Point to your `lora_output` directory
3. Select a checkpoint from the dropdown (e.g., `epoch_300`)
4. Start training — it continues from exactly where it left off

### Adapter Merge into Base Model

Merges the adapter weights (LoRA or LoKr) directly into the base model, producing a standalone safetensors file that doesn't need the adapter at inference time.

**Process:**
1. Go to the "Merge" section
2. Select the base model checkpoint (e.g., `acestep-v15-turbo`)
3. Select the adapter checkpoint (from `best/` or `final/` or any epoch)
4. Choose output directory (default: `checkpoints/acestep-v15-merged`)
5. Click "Merge" — the merge tab auto-detects the adapter type:
   - **LoRA**: Uses PEFT's `merge_and_unload()` to bake weights into the model
   - **LoKr**: Uses LyCORIS's `merge_to()` to apply Kronecker factors to the base weights

**Output:** A full model directory with `model.safetensors` (~4.5GB) and `silence_latent.pt` that can be used directly as a checkpoint.

---

## Captioner Deep Dive

### How Captioning Works

The captioner uses **Qwen2.5-Omni** (11B parameters), a multimodal model that understands audio. Audio is loaded at 16kHz mono, fed to the model with the prompt `"Describe this audio in detail"`, and the model generates a natural language description.

**Inference settings:**
- Temperature: 0.7 (creative but coherent)
- Top-p: 0.9
- Max tokens: 512

### How Metadata Extraction Works

For each audio file, metadata is extracted in two ways:

**Librosa analysis (local, no model needed):**
- BPM via beat tracking
- Duration from sample count

**AI analysis (via captioner model):**
- Musical key (e.g., "D Minor")
- Time signature (e.g., "4")
- Genre (e.g., "rock")

The model is prompted with a structured format request, and the output is parsed with cascading regex patterns to handle various model response formats.

### How Lyrics Transcription Works

The transcriber (`ACE-Step/acestep-transcriber`) is a separate Qwen2.5-Omni model fine-tuned specifically for structured lyrics extraction. It outputs lyrics with section tags:

```
[Verse 1]
Walking down the empty street tonight
The neon lights reflecting off the rain

[Chorus]
We are the ones who carry on
Through every storm we remain strong
```

**Recognized section tags:** `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`, `[Pre-Chorus]`, `[Post-Chorus]`, `[Instrumental]`, `[Spoken]`, `[Guitar]`, `[Piano]`, `[Interlude]`

If the model detects no lyrics (instrumental track), it outputs `[Instrumental]`.

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Use a lower GPU preset or enable Gradient Checkpointing + Encoder Offloading
- Reduce batch size to 1
- Reduce LoRA rank (32 → 16)
- Reduce max crop length (1500 → 1000)
- Use Adafactor instead of AdamW (much less optimizer state)
- Switch attention type to "self" only

**Checkpoint dropdown is empty**
- The trainer scans both `./checkpoints/` and `./ACE-Step-1.5/checkpoints/` (for Pinokio layout)
- Each checkpoint directory must contain a `config.json` file
- Directory name must start with `acestep-v15-`
- If using Pinokio, ensure the symlink from `ACE-Step-1.5/checkpoints/` to the shared drive is intact

**Loss is NaN or explodes**
- Reduce learning rate (try 5e-5 or 1e-5)
- Check that your preprocessed tensors aren't corrupted — try re-preprocessing
- If using Prodigy, it manages its own LR so this is rare

**Loss plateaus very early and doesn't decrease**
- Try increasing learning rate
- Increase LoRA rank for more capacity
- Check that your dataset labels are meaningful (not all empty or identical)
- Make sure activation tag position is correct

**Training is very slow**
- Disable gradient checkpointing if you have enough VRAM (saves ~30% time)
- Increase batch size if VRAM allows
- Enable `torch.compile` (experimental, may not work on all systems)
- Use `num_workers > 0` on Linux (Windows works best with 0)

**Captioner produces poor descriptions**
- Ensure audio quality is reasonable (not overly compressed or noisy)
- The captioner works best with clear, well-mixed audio
- Consider writing or editing captions manually for best results

### Checking Logs

Training logs are saved to `lora_output/<project>/logs/`. Check these if training behaves unexpectedly.

---

## Security Notes

This section covers security considerations for running the trainer. The tool runs locally on your machine and processes local files, but it's worth understanding the trust boundaries.

### Network Access

| Component | Network Access | What It Connects To |
|-----------|---------------|---------------------|
| Model auto-download | Outbound HTTPS | `huggingface.co`, `modelscope.cn` |
| Gradio UI | Local only | `127.0.0.1:7861` (never exposed externally unless `--share` is used) |
| Training loop | None | Fully offline after model download |
| pip/uv install | Outbound HTTPS | `pypi.org` (during install only) |

### `--share` Flag Warning

Using `python launch.py --share` creates a public Gradio tunnel (via `gradio.live`). This exposes your UI to the internet. **Do not use `--share` on machines with sensitive data.** The default (`127.0.0.1`) is local-only and safe.

### Allowed File Paths

The Gradio server is configured with broad allowed paths (`/`, `C:\`, `D:\`, etc.) to support file browsing. This is necessary for the file picker functionality but means the Gradio server can serve any local file. This is safe when running locally, but avoid exposing the server externally.

### Dependencies

The project depends on several third-party packages from PyPI. Notable trust considerations:

- **PyTorch, Transformers, PEFT, Accelerate** — maintained by Meta/HuggingFace, widely audited
- **Gradio** — maintained by HuggingFace, widely used
- **prodigyopt** — smaller package for the Prodigy optimizer, check version pinning
- **bitsandbytes** — maintained by Tim Dettmers, widely used in ML community
- **modelscope** — Alibaba's model hub SDK, used as fallback download source

All dependencies are pinned to minimum versions in `requirements.txt`. Review the dependency list if you have specific security requirements.

### Tensor Files

Preprocessed `.pt` files are PyTorch tensors saved with `torch.save()`. PyTorch's default pickle-based serialization can execute arbitrary code if the `.pt` file is malicious. **Only load `.pt` files you created yourself or trust.** The `.safetensors` format used for model weights and LoRA adapters is safe by design (no code execution).

### Model Weights

Model weights are downloaded from HuggingFace Hub or ModelScope with checksum verification. The `model_downloader.py` includes MD5 checksum validation for integrity verification.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) — Base music generation model
- [Side-Step](https://github.com/koda-dernet/Side-Step) — Inspiration for advanced training features
- Built with [Claude Code](https://claude.ai)
