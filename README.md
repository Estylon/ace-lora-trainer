# ACE-Step LoRA Trainer + Captioner

A standalone Gradio UI for training LoRA adapters on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) music generation models, with an integrated audio captioner.

## Features

### LoRA Training
- **GPU Presets** — One-click optimal settings for RTX 4090, 3090, 3080, 3060
- **Multiple Optimizers** — AdamW, AdamW 8-bit, Adafactor, Prodigy (auto-LR)
- **Multiple Schedulers** — Cosine, Linear, Constant, Constant+Warmup
- **Attention Targeting** — Train self-attention, cross-attention, or both
- **Gradient Checkpointing** — Save ~40-60% VRAM at ~30% speed cost
- **Encoder Offloading** — Move encoder to CPU during training, save ~2-4GB VRAM
- **Gradient Sensitivity Estimation** — Identify which layers matter most for your dataset
- **Random Crop Augmentation** — Full songs randomly cropped to 60s each epoch (faster training, natural augmentation)
- **Early Stopping** — MA5 smoothed loss tracking with configurable patience
- **Auto-Save Best Model** — Automatically saves the checkpoint with lowest loss
- **Resume Training** — Continue from any saved checkpoint
- **LoRA Merge** — Merge LoRA into base model as standalone safetensors

### Audio Captioner
- **AI Captioning** — Automatic music style description using Qwen2.5 Omni
- **Lyrics Transcription** — Structured lyrics with [Verse], [Chorus] tags
- **Metadata Extraction** — BPM, key, time signature, duration, genre
- **Batch Processing** — Process entire directories of audio files

## Quick Start

### 1. Install

```bash
# Clone this repo
git clone https://github.com/Estylon/ace-lora-trainer.git
cd ace-lora-trainer

# Create virtual environment
python -m venv env
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# Or: pip install -r requirements.txt
```

### 2. Models

**ACE-Step (generation model)** — Auto-downloaded on first launch to `./checkpoints/`.

**Captioner & Transcriber** — Auto-downloaded from HuggingFace on first use:
- Captioner: `ACE-Step/acestep-captioner` (~22GB)
- Transcriber: `ACE-Step/acestep-transcriber` (~22GB)

### 3. Launch

```bash
# Training UI (default)
python launch.py

# Captioner UI
python launch.py --mode caption

# Both UIs simultaneously
python launch.py --mode both
```

## Recommended Settings by GPU

| GPU | VRAM | Batch | Rank | Optimizer | Scheduler | VRAM Features |
|-----|------|-------|------|-----------|-----------|---------------|
| RTX 4090 / 5090 | 24GB+ | 3 | 64 | Prodigy | Cosine | None needed |
| RTX 3090 / 4080 | 16-24GB | 2 | 64 | Prodigy | Cosine | None needed |
| RTX 3080 / 4070 | 10-12GB | 1 | 32 | AdamW 8bit | Cosine | Grad Ckpt + Offload |
| RTX 3060 / 4060 | 8GB | 1 | 16 | Adafactor | Constant+Warmup | Grad Ckpt + Offload |

> **Note (8GB):** With only 8GB VRAM, AdamW 8-bit may still OOM. Adafactor stores nearly zero optimizer state, making it the safest choice for low-VRAM GPUs.

## Training Workflow

1. **Load Model** — Select and initialize an ACE-Step checkpoint
2. **Build Dataset** — Scan audio files, auto-label with AI captioner
3. **Preprocess** — Convert audio + labels to training tensors
4. **Train** — Select GPU preset or configure manually, start training
5. **Export** — Export LoRA or merge into base model

## Epoch Recommendations

| Songs | Recommended Epochs | Save Every |
|-------|-------------------|------------|
| 1-3 | 1500 | 200 |
| 4-6 | 1000 | 200 |
| 7-10 | 700 | 100 |
| 11-20 | 500 | 100 |
| 21-50 | 300 | 50 |
| 50+ | 200 | 50 |

With Early Stop enabled, training will automatically stop when loss plateaus.

## Project Structure

```
ace-lora-trainer/
├── launch.py                 # Main launcher
├── lora_training_ui.py       # Training Gradio UI
├── captioner_standalone.py   # Captioner Gradio UI
├── requirements.txt          # Python dependencies
├── acestep/                  # Core ACE-Step modules
│   ├── training/             # Training pipeline
│   │   ├── trainer.py        # LoRA trainer (Fabric + basic)
│   │   ├── configs.py        # Training configuration
│   │   ├── data_module.py    # Data loading
│   │   ├── dataset_builder.py# Dataset management
│   │   ├── lora_utils.py     # LoRA injection/save/load
│   │   └── estimator.py      # Gradient sensitivity estimation
│   ├── handler.py            # Model handler
│   ├── llm_inference.py      # LLM for captioning
│   └── ...                   # Other ACE-Step modules
├── checkpoints/              # Model weights (auto-downloaded)
├── datasets/                 # Training datasets
└── lora_output/              # Training output
```

## Documentation

For a complete, in-depth guide covering every parameter, feature, and workflow:

**[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** — Full training guide with parameter explanations, GPU presets, disk space requirements, security notes, and troubleshooting.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) — Base music generation model
- [Side-Step](https://github.com/koda-dernet/Side-Step) — Inspiration for advanced training features
- Built with [Claude Code](https://claude.ai)
