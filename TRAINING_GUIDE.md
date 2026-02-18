# ACE-Step LoRA Training Guide

A complete guide to training LoRA and LoKr adapters on ACE-Step 1.5 music generation models.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Disk Space Requirements](#disk-space-requirements)
5. [The Two Interfaces](#the-two-interfaces)
6. [Complete Training Workflow](#complete-training-workflow)
   - [Step 1: Prepare Your Audio (Captioning)](#step-1-prepare-your-audio-captioning)
   - [Step 2: Initialize the Service](#step-2-initialize-the-service)
   - [Step 3: Build Your Dataset](#step-3-build-your-dataset)
   - [Step 4: Preprocess Tensors](#step-4-preprocess-tensors)
   - [Step 5: Configure Training](#step-5-configure-training)
   - [Step 6: Train](#step-6-train)
   - [Step 7: Export or Merge](#step-7-export-or-merge)
7. [Understanding the Parameters](#understanding-the-parameters)
   - [Adapter Type: LoRA vs LoKr](#adapter-type-lora-vs-lokr)
   - [Optimizers](#optimizers)
   - [Schedulers](#schedulers)
   - [Attention Targeting](#attention-targeting)
   - [VRAM Saving Features](#vram-saving-features)
   - [Loss, Early Stopping & Best Model](#loss-early-stopping--best-model)
   - [Random Crop Augmentation](#random-crop-augmentation)
   - [Turbo vs Base Model Training](#turbo-vs-base-model-training)
8. [GPU Presets Reference](#gpu-presets-reference)
9. [Epoch & Dataset Guidelines](#epoch--dataset-guidelines)
10. [Labeling Your Audio: Three Approaches](#labeling-your-audio-three-approaches)
11. [Advanced Features](#advanced-features)
    - [Gradient Sensitivity Estimation](#gradient-sensitivity-estimation)
    - [Resume from Checkpoint](#resume-from-checkpoint)
    - [LoRA Merge into Base Model](#lora-merge-into-base-model)
    - [Split Long Audio Files](#split-long-audio-files)
12. [Captioner Deep Dive](#captioner-deep-dive)
13. [Troubleshooting](#troubleshooting)
14. [Security Notes](#security-notes)

---

## Overview

This tool lets you fine-tune ACE-Step 1.5 music generation models using **LoRA** (Low-Rank Adaptation) or **LoKr** (Low-Rank Kronecker, via LyCORIS). Instead of retraining the entire 4.5GB model, these adapters train small weights (43-85MB) that modify the model's behavior:

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
| GPU | 8GB VRAM (RTX 3060) | 24GB VRAM (RTX 4090/5090) |
| RAM | 16GB | 32GB |
| Storage | 30GB free | 60GB+ free |
| OS | Windows 10/11, Linux, macOS | Windows 11, Ubuntu 22.04+ |

### Software

- Python 3.10 or 3.11 (3.13 also works)
- CUDA 12.1+ (for NVIDIA GPUs)
- Git

---

## Installation

```bash
git clone https://github.com/Estylon/ace-lora-trainer.git
cd ace-lora-trainer

# Create virtual environment (uv preferred, or python fallback)
uv venv env          # If uv is installed
# python -m venv env # Fallback if uv is not installed

# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# Install dependencies (uv is faster, pip works too)
uv pip install -r requirements.txt
# Or: pip install -r requirements.txt
```

Or use the one-click installers: `install.bat` (Windows) / `install.sh` (Linux/Mac).

### Optional: 8-bit Optimizer Support

For AdamW 8-bit (saves ~30-40% optimizer VRAM):

```bash
# Windows/Linux only (not supported on macOS)
uv pip install bitsandbytes>=0.43.0
```

### Models (Auto-Download)

All models auto-download from HuggingFace on first use. No manual download needed.

| Model | Size | What It Does |
|-------|------|--------------|
| ACE-Step v1.5 (DiT + VAE + text encoder) | ~6GB total | The generation model you'll fine-tune |
| 5Hz LM 1.7B | ~3.5GB | AI captioning/labeling inside the trainer (optional) |
| ACE-Step Captioner (Qwen2.5-Omni) | ~22GB | Standalone high-quality audio captioner |
| ACE-Step Transcriber (Qwen2.5-Omni) | ~22GB | Structured lyrics extraction |

---

## Disk Space Requirements

### Per-Song Preprocessed Tensors

| Song Duration | Tensor Size |
|---------------|-------------|
| 3 minutes | ~4.5 MB |
| 4 minutes | ~4.8 MB |
| 5 minutes | ~5.1 MB |

**Rule of thumb: ~5 MB per song.**

### Checkpoint Sizes

| LoRA Rank | Adapter Only (best/final) | Full Checkpoint (adapter + optimizer state) |
|-----------|---------------------------|---------------------------------------------|
| 16 | ~11 MB | ~32 MB |
| 32 | ~43 MB | ~128 MB |
| 64 | ~85 MB | ~253 MB |

### Total Disk Space Estimates

| Scenario | Epochs | Save Every | Rank | Checkpoints Created | Total Disk |
|----------|--------|------------|------|---------------------|------------|
| **Small dataset, 4090** | 800 | 50 | 64 | 16 regular + best + final | **~4.2 GB** |
| **Small dataset, 3080** | 1000 | 50 | 32 | 20 regular + best + final | **~2.6 GB** |
| **Small dataset, 3060** | 1000 | 50 | 16 | 20 regular + best + final | **~0.7 GB** |
| **Large dataset, 4090** | 300 | 50 | 64 | 6 regular + best + final | **~1.7 GB** |

### Summary: Total Project Disk Usage

| What | Space |
|------|-------|
| This repo (code) | ~2 MB |
| Python venv + dependencies | ~8-12 GB |
| ACE-Step model (auto-downloaded) | ~6 GB |
| 5Hz LM (optional, for AI labeling) | ~3.5 GB |
| Captioner model (optional, standalone) | ~22 GB |
| Transcriber model (optional) | ~22 GB |
| Training tensors (10 songs) | ~50 MB |
| Training checkpoints (typical run) | ~1-4 GB |
| **Total (trainer only)** | **~16-22 GB** |
| **Total (with captioner + transcriber)** | **~58-66 GB** |

> **Tip:** Delete old checkpoints from `lora_output/<project>/checkpoints/` once you've identified the best epoch. Keep only `best/` and `final/`.

---

## The Two Interfaces

This tool ships two **separate** Gradio web UIs:

### Captioner UI — `python launch.py --mode caption`
A standalone captioner for high-quality audio analysis:
- **Captions** — Detailed music style descriptions (genre, instruments, structure, mood)
- **Metadata** — BPM, musical key, time signature, duration, genre
- **Lyrics** — Structured lyrics with `[Verse]`, `[Chorus]`, `[Bridge]` section tags

### Trainer UI — `python launch.py --mode train`
The full training pipeline:
- Initialize the ACE-Step model
- Build and edit your dataset (with optional built-in AI labeling)
- Preprocess audio into training tensors
- Configure and run training
- Export the trained adapter

### Launch Modes

```bash
python launch.py                  # Training UI only (port 7861)
python launch.py --mode caption   # Captioner UI only (port 7861)
python launch.py --mode both      # Both UIs (trainer: 7861, captioner: 7862)
python launch.py --port 8080      # Custom port
python launch.py --share          # Create public Gradio link
```

> **Note:** `--mode both` launches the captioner as a separate process for stability. Both UIs run independently and can be used simultaneously.

---

## Complete Training Workflow

### Step 1: Prepare Your Audio (Captioning)

**Goal:** Generate text descriptions and metadata for each audio file. The model needs text labels to learn what it's hearing.

You have **three labeling approaches** (see [Labeling Your Audio](#labeling-your-audio-three-approaches) for details):

| Approach | Quality | Effort | VRAM Needed |
|----------|---------|--------|-------------|
| **Standalone Captioner** (recommended) | Best | Low | ~22GB per model |
| **Built-in AI Label** (convenient) | Good | Lowest | ~3.5GB extra |
| **Manual / CSV** | Depends on you | Highest | None |

**Recommended approach — Standalone Captioner:**

1. Launch: `python launch.py --mode caption`
2. Set **Activation Tag** (e.g., `ZX_MyArtist`) — a unique trigger word for your style
3. Enter the path to your audio folder
4. Click **"Load Captioner"** — model (~22GB) downloads automatically on first use
5. Click **"Caption All"** — generates for each file:
   - Detailed text caption
   - BPM, musical key, time signature, genre
6. Optionally load the Transcriber for structured lyrics
7. Each audio file gets a `.json` sidecar file with all metadata

**VRAM-saving two-pass workflow (for < 24GB VRAM):**
1. Load Captioner → Caption All → **Unload** (frees ~22GB)
2. Load Transcriber → Transcribe Only → **Unload**

**Output JSON format:**
```json
{
  "filename": "song_01.mp3",
  "caption": "A high-energy rock track with distorted electric guitar riffs...",
  "bpm": 142,
  "keyscale": "E Minor",
  "timesignature": "4",
  "genre": "rock",
  "lyrics": "[Verse 1]\nWalking down the empty street...",
  "language": "en",
  "is_instrumental": false,
  "duration": 234,
  "custom_tag": "ZX_MyArtist"
}
```

### Step 2: Initialize the Service

1. Launch the trainer: `python launch.py`
2. Open http://127.0.0.1:7861
3. In **"1. Service"**, select a checkpoint (e.g., `acestep-v15-turbo`)
4. Click **"Initialize Service"**

The model downloads automatically on first use (~6GB). The trainer uses **lazy loading**: only the DiT model loads initially (~4.5GB). The VAE and text encoder load on-demand when needed (preprocessing, auto-labeling).

The trainer auto-detects **turbo** vs **base** model and configures parameters accordingly.

### Step 3: Build Your Dataset

1. In **"2. Dataset"**, enter the path to your audio folder (the same folder where the captioner saved JSONs)
2. Set a **Dataset Name** (e.g., `my_artist_lora`)
3. Set an **Activation Tag** — must match what you used in captioning (e.g., `ZX_MyArtist`)
4. Choose **Tag Position**:
   - `prepend` (recommended) — adds the tag before the caption: `"ZX_MyArtist, A high-energy rock track..."`
   - `replace` — replaces caption entirely with just the tag
   - `append` — adds the tag after the caption
5. Click **"Scan Audio Directory"** — finds audio files and loads any existing `.json` labels
6. Review and edit individual samples if needed

**Optional: Built-in AI labeling** — If you didn't use the standalone captioner, you can click **"Download & Enable AI Labeling (~3.5 GB)"** to download the 5Hz LM model, then click **"Auto-Label All"** to generate labels directly in the trainer.

### Step 4: Preprocess Tensors

1. In **"3. Preprocess"**, verify the tensor output directory
2. Set **Max Duration** (default 240s) — longer songs get truncated
3. Click **"Preprocess Tensors"**

This encodes each audio through the VAE and tokenizes captions with the text encoder, producing `.pt` tensor files (~5MB each). The VAE and text encoder load automatically if not already loaded.

> **Note:** Preprocessing is a one-time step. You can reuse the same tensors across multiple training runs with different hyperparameters.

### Step 5: Configure Training

**Quick setup — GPU Presets:**
1. Select your GPU tier from the dropdown (e.g., "RTX 4090 / 5090")
2. All parameters are auto-configured optimally for your VRAM
3. Presets are **adapter-aware**: selecting LoRA or LoKr applies different optimal settings

**Or configure manually** — see [Understanding the Parameters](#understanding-the-parameters) for details.

### Step 6: Train

1. Click **"Start Training"**
2. Watch the live loss plot — loss should decrease over time
3. The log window shows epoch progress, loss, learning rate, and checkpoint saves
4. Training ends when:
   - Max epochs reached, or
   - Early stopping triggers (if enabled), or
   - You click "Stop Training"

**What happens during training:**
- Each epoch iterates over your entire dataset
- Audio tensors are randomly cropped to 60s windows each epoch (data augmentation)
- The model learns to predict the noise flow field, minimizing MSE loss
- Checkpoints saved at regular intervals to `lora_output/<project>/checkpoints/`
- The best model (lowest MA5 smoothed loss) saved to `lora_output/<project>/best/`
- **Final weights = best MA5 checkpoint** — when training completes, the `best/` checkpoint is copied as `final/`, ensuring you always get the best result

### Step 7: Export or Merge

**Option A: Use the adapter directly (recommended)**
- The adapter in `best/adapter/` or `final/adapter/` can be loaded at inference time
- Small file size (~43-85MB depending on rank)
- Works for both LoRA and LoKr adapters

**Option B: Merge adapter into base model**
- Creates a standalone model with the adapter baked in (~4.5GB)
- No need to load the adapter separately
- Go to **"Merge"**, select base model + adapter checkpoint, click **"Merge"**
- Auto-detects adapter type (LoRA or LoKr) from checkpoint files

---

## Understanding the Parameters

### Adapter Type: LoRA vs LoKr

| | LoRA (default) | LoKr (experimental) |
|---|---|---|
| **Library** | PEFT (HuggingFace) | LyCORIS |
| **Method** | Low-Rank decomposition (A x B) | Kronecker product factorization |
| **Maturity** | Proven, widely used | Experimental, fewer users |
| **File format** | `adapter_config.json` + `adapter_model.safetensors` | `lokr_config.json` + `lokr_weights.safetensors` |

Both adapters target the same attention layers (q/k/v/o projections in every DiT attention block) and use the same training loop. The GPU presets automatically apply different optimal hyperparameters for each adapter type.

**Key differences in LoKr training:**
- Higher learning rate (3e-4 vs 1e-4 for LoRA) — Kronecker factorization benefits from stronger gradients
- More gradient accumulation (4-8x vs 1-2x) — smooths noisy gradients
- No dropout needed (0.0 vs 0.1) — implicit regularization from Kronecker structure
- AdamW instead of Prodigy — more predictable with LoKr

### LoRA Parameters

| Parameter | Default | Range | What It Does |
|-----------|---------|-------|--------------|
| **Rank (r)** | 64 | 4-256 | Capacity of the adapter. 16 for subtle style, 32 for solid training, 64 for maximum fidelity. |
| **Alpha** | 128 | 4-512 | Scaling factor. Common practice: set to 2x rank. |
| **Dropout** | 0.1 | 0.0-0.5 | Regularization. Increase to 0.15-0.2 if overfitting on small datasets. |

### LoKr Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| **Factor** | -1 (auto) | Kronecker factor. -1 = automatic (sqrt of dimension). |
| **Linear Dim** | 10000 (auto) | Rank for the linear component. 10000 = auto-selection. |
| **Linear Alpha** | 1.0 | Scaling factor for the LoKr adapter. |
| **Decompose Both** | Off | Applies Kronecker decomposition to both factors (more expressive). |
| **Use Tucker** | Off | Uses Tucker decomposition for finer factorization. |
| **Dropout** | 0.0 | Dropout for LoKr layers. |

### Optimizers

| Optimizer | Best For | Memory | How It Works |
|-----------|----------|--------|--------------|
| **AdamW** | General training | High (~2x model params) | Standard optimizer with weight decay. Reliable default. |
| **AdamW 8-bit** | 10-16GB VRAM | Medium (~1.2x model params) | Same as AdamW, stores momentum in 8-bit. Requires `bitsandbytes`. |
| **Adafactor** | 8GB VRAM | Very Low (~0.01x model params) | Near-zero optimizer state. Best when VRAM is critically limited. |
| **Prodigy** | 16GB+ VRAM | High (~3x model params) | Auto-tunes its own learning rate. Set LR to any value — Prodigy finds the optimal rate itself. Automatically uses constant scheduler. |

### Schedulers

All schedulers include a warmup phase — LR ramps up linearly from 10% to 100% over the first N steps.

| Scheduler | After Warmup | Best Paired With |
|-----------|-------------|------------------|
| **Cosine** | LR follows a cosine curve down to 1% | AdamW, AdamW 8-bit |
| **Linear** | LR decreases linearly to 1% | AdamW, AdamW 8-bit |
| **Constant** | LR stays at 100% | Prodigy (forced) |
| **Constant + Warmup** | LR ramps up then stays at 100% | Adafactor, Prodigy |

### Attention Targeting

| Mode | What Gets Trained | When To Use |
|------|-------------------|-------------|
| **Both** (default) | Self-attention + Cross-attention | Maximum expressiveness. 16GB+ VRAM. |
| **Self** | Self-attention only | Saves ~40% adapter params. 8GB GPUs or subtle style transfer. |
| **Cross** | Cross-attention only | Trains text conditioning response. Experimental. |

### VRAM Saving Features

| Feature | VRAM Saved | Speed Cost | How It Works |
|---------|-----------|------------|--------------|
| **Gradient Checkpointing** | ~40-60% | ~30% slower | Recomputes activations during backward pass instead of storing them. |
| **Encoder Offloading** | ~2-4 GB | Minimal | Moves text encoder, VAE, tokenizer to CPU during training. |
| **Self-attn Only** | ~40% adapter | None | Simply trains fewer parameters. |
| **Lower Rank** | Proportional | None | Rank 16 uses ~4x less memory than rank 64. |

**Recommended stacking for low VRAM:**
- **8GB:** Grad Checkpointing + Encoder Offloading + Self-attn only + Rank 16 + Adafactor + Batch 1
- **10-12GB:** Grad Checkpointing + Encoder Offloading + Both attn + Rank 32 + AdamW 8-bit + Batch 1
- **16-24GB:** No VRAM saving needed. Prodigy + Rank 64 + Batch 2-3.

### Loss, Early Stopping & Best Model

**Flow Matching Loss:** The model learns by predicting the "flow" between noise and data. Loss = MSE between predicted and actual flow. Lower = better.

**Best Model Tracking (MA5):**
- Activates after the **Auto-Save Best After** warmup (default: 200 epochs for LoRA, 100 for LoKr)
- Uses a **Moving Average over 5 epochs (MA5)** to smooth fluctuations
- Saves a new best only if `smoothed_loss < best_loss - 0.001`
- Best model saved to `output_dir/best/adapter/`
- **Final model = best MA5 checkpoint** — at training completion, `best/` is copied as `final/`

**Early Stopping:**
- Activates after the same warmup period
- Counts epochs without a new best (patience counter)
- When `patience_counter >= patience_value`, training stops
- Default patience: 80 (LoRA) / 50 (LoKr)

**Why the warmup?** The first 100-200 epochs are volatile. Tracking too early gives false signals.

### Random Crop Augmentation

If your songs are longer than the crop length (default: 1500 frames = 60s), each epoch randomly selects a different window. The model sees different parts of the same song — natural data augmentation.

**Frame rate:** 25 frames/second. 1500 frames = 60s, 3000 frames = 120s.

| Setting | Duration | When To Use |
|---------|----------|-------------|
| 1500 (default) | 60s | Standard training |
| 1000 | 40s | Low VRAM |
| 3000 | 120s | More context, more VRAM |
| 0 | Full song | No cropping (high VRAM) |

### Turbo vs Base Model Training

Both model types use logit-normal timestep sampling and CFG dropout. Parameters are read from each model's `config.json`.

| Aspect | Turbo | Base / SFT |
|--------|-------|------------|
| **Inference speed** | Fast (8 steps) | Slower (60 steps), higher quality |
| **Inference shift** | 3.0 | 1.0 |

**When to use which:**
- **Turbo** — faster iteration, good for style transfer, most common choice
- **Base** — maximum quality, fully supported

---

## GPU Presets Reference

Presets are **adapter-aware** — they apply different optimal settings when LoRA or LoKr is selected.

### RTX 4090 / 5090 (24GB+)

| Parameter | LoRA | LoKr |
|-----------|------|------|
| Rank | 64 | 64 |
| Alpha | 128 | 128 |
| Dropout | 0.1 | 0.0 |
| Learning Rate | 1e-4 | 3e-4 |
| Max Epochs | 800 | 500 |
| Batch Size | 3 | 2 |
| Grad Accumulation | 1 | 4 |
| Optimizer | Prodigy | AdamW |
| Scheduler | Cosine | Cosine |
| Early Stop Patience | 80 | 50 |
| Best After | 200 | 100 |

### RTX 3090 / 4080 (16-24GB)

Same as 4090 except Batch Size: **2** (LoRA) / **2** (LoKr), Grad Accumulation: **2** (LoRA) / **4** (LoKr).

### RTX 3080 / 4070 (10-12GB)

| Parameter | LoRA | LoKr |
|-----------|------|------|
| Rank | 32 | 32 |
| Optimizer | AdamW 8-bit | AdamW |
| Grad Checkpointing | On | On |
| Encoder Offloading | On | On |
| Batch Size | 1 | 1 |
| Grad Accumulation | 4 | 8 |

### RTX 3060 / 4060 (8GB)

| Parameter | LoRA | LoKr |
|-----------|------|------|
| Rank | 16 | 16 |
| Optimizer | Adafactor | Adafactor |
| Scheduler | Constant+Warmup | Constant+Warmup |
| Attention | Self only | Self only |
| Grad Checkpointing | On | On |
| Encoder Offloading | On | On |
| Max Crop Length | 1000 (40s) | 1000 (40s) |

---

## Epoch & Dataset Guidelines

| Number of Songs | Recommended Epochs | Save Every | Expected Time (4090) |
|----------------|-------------------|------------|----------------------|
| 1-3 | 1500 | 200 | ~30-45 min |
| 4-6 | 1000 | 200 | ~30-50 min |
| 7-10 | 700 | 100 | ~40-60 min |
| 11-20 | 500 | 100 | ~45-90 min |
| 21-50 | 300 | 50 | ~60-120 min |
| 50+ | 200 | 50 | Varies |

**Tips:**
- More songs = fewer epochs needed
- Random crop augmentation multiplies your dataset — a 5-minute song has ~5 possible 60s windows
- The best checkpoint is not always the last one — `final/` always contains the best MA5 checkpoint
- Loss oscillating? Decrease learning rate or increase batch size/grad accumulation
- Loss plateauing early? Increase learning rate or rank

---

## Labeling Your Audio: Three Approaches

### Approach 1: Standalone Captioner (recommended for quality)

The standalone captioner (`python launch.py --mode caption`) uses Qwen2.5-Omni (11B), a full multimodal model that produces detailed, accurate music descriptions. It saves `.json` sidecar files next to each audio file.

**Pros:** Highest quality captions, extracts key/genre/BPM, structured lyrics
**Cons:** Requires ~22GB VRAM per model (captioner + transcriber loaded separately)

### Approach 2: Built-in AI Labeling (convenient, good quality)

The trainer's Dataset tab includes a **"Download & Enable AI Labeling"** button that downloads the 5Hz LM model (~3.5GB). This smaller model generates captions directly inside the trainer.

**Pros:** No separate UI, automatic download, lower VRAM (~3.5GB extra)
**Cons:** Lower quality than the standalone captioner

**How to use:**
1. Initialize the service first (Step 1)
2. Scan your audio directory (Step 3)
3. Click **"Download & Enable AI Labeling (~3.5 GB)"** — downloads once, loads into memory
4. Click **"Auto-Label All"**

### Approach 3: Manual / CSV Import (full control)

Write your own captions and metadata, either directly in the sample editor or by preparing a CSV file that gets loaded alongside your audio.

**Pros:** Full control over every label, no VRAM needed
**Cons:** Time-consuming for large datasets

**CSV format:** Place a CSV in your audio directory with columns: `filename`, `caption`, `genre`, `bpm`, `keyscale`, `timesignature`, `lyrics`, `language`

---

## Advanced Features

### Gradient Sensitivity Estimation

Estimates which attention layers are most sensitive to your dataset before training.

1. Load model and preprocess tensors first
2. Go to the Gradient Estimation accordion
3. Set number of batches (10 is good)
4. Click "Run Estimation"
5. Review the ranked table — score 0.0 to 1.0 relative to the most sensitive module

### Resume from Checkpoint

1. Enable "Resume from Checkpoint" in the training section
2. Point to your `lora_output` directory
3. Select a checkpoint (e.g., `epoch_300`)
4. Start training — continues from exactly where it left off (optimizer state, scheduler, epoch)

### LoRA Merge into Base Model

Merges adapter weights into the base model for standalone inference:

1. Go to **"Merge"** section
2. Select base model + adapter checkpoint
3. Click **"Merge"** — auto-detects LoRA vs LoKr
4. Output: `model.safetensors` (~4.5GB) + `silence_latent.pt`

### Split Long Audio Files

The trainer includes an audio splitter (in the Dataset tab) that cuts long songs into shorter segments for faster training:

- Uses `torchaudio` natively — **no ffmpeg required**
- Supports mp3, wav, flac, ogg
- Recommended: 30s segments for fastest training, 60s for more context
- Segments shorter than 10s are automatically discarded

---

## Captioner Deep Dive

### How Captioning Works

The standalone captioner uses **Qwen2.5-Omni** (11B parameters). Audio is loaded at 16kHz mono, fed to the model, and generates a natural language description.

**Inference settings:** Temperature 0.7, Top-p 0.9, Max tokens 512

### How Metadata Extraction Works

**Librosa analysis (fast, no model needed):**
- BPM via beat tracking
- Duration from sample count

**AI analysis (via captioner model):**
- Musical key (e.g., "D Minor")
- Time signature (e.g., "4")
- Genre (e.g., "electronic pop")

The model is prompted with a structured format. Output is parsed with cascading regex patterns. If the structured extraction fails, the caption text itself is mined for key/genre mentions.

### How Lyrics Transcription Works

The transcriber (`ACE-Step/acestep-transcriber`) outputs structured lyrics with section tags:

```
[Verse 1]
Walking down the empty street tonight

[Chorus]
We are the ones who carry on
```

If no lyrics are detected, it outputs `[Instrumental]`.

### Batch Processing & VRAM

When captioning large batches (50+ files):
- Audio is loaded once per file and reused for caption + metadata
- GPU cache is cleared between files to prevent OOM
- JSON files are saved incrementally (one per file), so progress is preserved if interrupted

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Use a lower GPU preset
- Enable Gradient Checkpointing + Encoder Offloading
- Reduce batch size to 1
- Reduce LoRA rank (32 -> 16)
- Use Adafactor instead of AdamW
- Switch attention to "self" only

**"Model not initialized" during auto-labeling**
- Click "Initialize Service" first (Step 1) before using any features
- The VAE and text encoder load automatically when needed

**"LLM not available" on Auto-Label**
- Click "Download & Enable AI Labeling (~3.5 GB)" first
- Or use the standalone captioner / CSV / manual labels instead

**Split audio gives `[WinError 2]`**
- This was a bug with older versions using pydub/ffmpeg. Update to the latest version which uses torchaudio (no ffmpeg needed).

**Checkpoint dropdown is empty**
- The trainer scans `./checkpoints/` and `./ACE-Step-1.5/checkpoints/`
- Each checkpoint directory must contain a `config.json` file
- Directory name must start with `acestep-v15-`

**Loss is NaN or explodes**
- Reduce learning rate (try 5e-5 or 1e-5)
- Re-preprocess tensors
- If using Prodigy, this is rare — check your dataset

**Loss plateaus early**
- Increase learning rate or LoRA rank
- Check that labels are meaningful (not all empty/identical)
- Verify activation tag position

**Training is very slow**
- Disable gradient checkpointing if VRAM allows
- Increase batch size
- Use `num_workers > 0` on Linux (Windows: keep at 0)

### Checking Logs

Training logs are saved to `lora_output/<project>/logs/`. Check these if training behaves unexpectedly.

---

## Security Notes

### Network Access

| Component | Network Access | Connects To |
|-----------|---------------|-------------|
| Model auto-download | Outbound HTTPS | `huggingface.co`, `modelscope.cn` |
| Gradio UI | Local only | `127.0.0.1:7861` (never exposed unless `--share`) |
| Training loop | None | Fully offline after download |

### `--share` Warning

`--share` creates a public Gradio tunnel. **Do not use on machines with sensitive data.** Default (`127.0.0.1`) is local-only.

### Tensor Files

`.pt` files use PyTorch's pickle-based serialization. **Only load `.pt` files you created or trust.** The `.safetensors` format (used for model weights and adapters) is safe by design.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) — Base music generation model
- [Side-Step](https://github.com/koda-dernet/Side-Step) — Inspiration for advanced training features
- Built with [Claude Code](https://claude.ai)
