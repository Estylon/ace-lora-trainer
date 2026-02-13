#!/usr/bin/env python3
"""
ACE-Step LoRA Training UI - Standalone Gradio Interface
Version 2.0 - UX Redesign

A lightweight, independent Gradio interface for LoRA training on ACE-Step.
Based on the Master Guide by Moonspell & AI Brother.

Usage:
    python lora_training_ui.py [--port 7861] [--share]
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import gradio as gr
from loguru import logger

# Standalone repo: acestep modules are at the same level as this file
ACESTEP_PATH = Path(__file__).parent

# Global handlers (initialized on demand)
dit_handler = None
llm_handler = None
current_model_type = "turbo"  # "turbo" or "base" — set during initialize_service

# Global state for workflow tracking
workflow_state = {
    "service_ready": False,
    "dataset_loaded": False,
    "dataset_labeled": False,
    "tensors_ready": False,
    "training_complete": False,
}


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory."""
    return ACESTEP_PATH / "checkpoints"


def list_available_checkpoints() -> List[str]:
    """List available model checkpoints that start with 'acestep-v15-'."""
    checkpoints_dir = get_checkpoints_dir()
    if not checkpoints_dir.exists():
        return []

    checkpoints = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and item.name.startswith("acestep-v15-"):
            if (item / "config.json").exists():
                checkpoints.append(item.name)

    return sorted(checkpoints)


# ============== Status Helper ==============

def get_status_html() -> str:
    """Generate status bar HTML based on current workflow state."""
    steps = [
        ("1. Service", workflow_state["service_ready"], "Load model"),
        ("2. Dataset", workflow_state["dataset_loaded"], "Scan audio"),
        ("3. Labels", workflow_state["dataset_labeled"], "Auto-label"),
        ("4. Tensors", workflow_state["tensors_ready"], "Preprocess"),
        ("5. Train", workflow_state["training_complete"], "Train LoRA"),
    ]

    html = '<div class="status-bar">'
    for name, done, hint in steps:
        status_class = "step-done" if done else "step-pending"
        icon = "✓" if done else "○"
        html += f'<div class="status-step {status_class}"><span class="step-icon">{icon}</span><span class="step-name">{name}</span></div>'
    html += '</div>'
    return html


def update_workflow_state(key: str, value: bool):
    """Update workflow state and return new status HTML."""
    workflow_state[key] = value
    return get_status_html()


# ============== File/Folder Picker Helpers ==============

def open_folder_picker() -> str:
    """Open a folder picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title="Select Folder")
        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        logger.warning(f"Folder picker failed: {e}")
        return ""


def open_file_picker(filetypes: List[Tuple[str, str]] = None) -> str:
    """Open a file picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if filetypes is None:
            filetypes = [("All files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        root.destroy()
        return file_path if file_path else ""
    except Exception as e:
        logger.warning(f"File picker failed: {e}")
        return ""


def open_save_file_picker(default_ext: str = ".json") -> str:
    """Open a save file dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        file_path = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=default_ext,
            filetypes=filetypes,
        )
        root.destroy()
        return file_path if file_path else ""
    except Exception as e:
        logger.warning(f"Save file picker failed: {e}")
        return ""


def pick_json_file() -> str:
    return open_file_picker([("JSON files", "*.json"), ("All files", "*.*")])


# ============== Service Functions ==============

def _detect_model_type(checkpoint_name: str) -> str:
    """Detect model type (turbo/base) from checkpoint name."""
    name_lower = checkpoint_name.lower()
    if "base" in name_lower:
        return "base"
    elif "sft" in name_lower:
        return "base"  # SFT models also use continuous timesteps
    return "turbo"


def initialize_service(checkpoint_name: str, progress=gr.Progress()):
    """Initialize ACE-Step service. Returns status + auto-detected model type params."""
    global dit_handler, llm_handler, workflow_state, current_model_type

    if not checkpoint_name:
        return "❌ Please select a checkpoint", get_status_html(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    progress(0.1, desc="Loading ACE-Step modules...")

    # Detect model type early for UI updates
    detected_type = _detect_model_type(checkpoint_name)
    current_model_type = detected_type

    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        progress(0.2, desc="Initializing DiT handler...")
        dit_handler = AceStepHandler()
        project_root = str(ACESTEP_PATH)

        progress(0.3, desc=f"Loading model: {checkpoint_name}...")
        status, success = dit_handler.initialize_service(
            project_root=project_root,
            config_path=checkpoint_name,
            device="auto",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )

        if not success:
            return f"❌ Failed to initialize DiT: {status}", get_status_html(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        progress(0.7, desc="Initializing LLM handler...")
        checkpoint_dir = str(get_checkpoints_dir())

        # Try 4B model first, then 1.7B
        lm_models_to_try = ["acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B"]
        llm_model_found = None

        for lm_model in lm_models_to_try:
            lm_model_path = os.path.join(checkpoint_dir, lm_model)
            weights_path = os.path.join(lm_model_path, "model-00001-of-00002.safetensors")
            if os.path.exists(weights_path):
                llm_model_found = lm_model
                break

        llm_info = ""
        if llm_model_found:
            llm_handler = LLMHandler()
            progress(0.8, desc=f"Loading LLM: {llm_model_found}...")
            llm_status, llm_success = llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=llm_model_found,
                backend="pt",
                device="auto",
                offload_to_cpu=False,
            )
            llm_info = f"✅ LLM: {llm_model_found}" if llm_success else f"⚠️ LLM failed"
        else:
            llm_handler = None
            llm_info = "⚠️ LLM not available (use CSV/manual labels)"

        progress(1.0, desc="Done!")
        workflow_state["service_ready"] = True

        device = dit_handler.device if dit_handler else "unknown"
        type_label = "⚡ Turbo" if detected_type == "turbo" else "🎯 Base"
        status_msg = f"✅ Ready | {type_label} | Model: {checkpoint_name} | Device: {device} | {llm_info}"

        # Auto-set training parameters based on model type
        if detected_type == "base":
            return (
                status_msg,
                get_status_html(),
                gr.update(value=1.0),         # shift → 1.0 for base
                gr.update(value="base"),       # model_type_radio
                gr.update(value=7.0),          # guidance_scale
                gr.update(value=0.1),          # cfg_dropout_prob
                gr.update(value=60, minimum=20),  # num_inference_steps
                gr.update(visible=True),       # base_params_group visibility
            )
        else:
            return (
                status_msg,
                get_status_html(),
                gr.update(value=3.0),          # shift → 3.0 for turbo
                gr.update(value="turbo"),       # model_type_radio
                gr.update(value=0.0),          # guidance_scale
                gr.update(value=0.1),          # cfg_dropout_prob
                gr.update(value=8, minimum=1),  # num_inference_steps
                gr.update(visible=False),      # base_params_group visibility
            )

    except Exception as e:
        logger.exception("Failed to initialize service")
        return f"❌ Error: {str(e)}", get_status_html(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def unload_service() -> Tuple[str, str]:
    """Unload service to free memory."""
    global dit_handler, llm_handler, workflow_state
    import gc
    import torch

    if dit_handler is not None:
        del dit_handler
        dit_handler = None
    if llm_handler is not None:
        del llm_handler
        llm_handler = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    workflow_state["service_ready"] = False
    return "✅ Service unloaded, GPU memory freed", get_status_html()


# ============== Audio Split Functions ==============

def split_audio_files(
    input_dir: str,
    output_dir: str,
    segment_duration: int,
    progress=gr.Progress(),
) -> str:
    """Split audio files into shorter segments for faster training.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save split segments
        segment_duration: Duration of each segment in seconds (30 or 60)

    Returns:
        Status message
    """
    if not input_dir or not input_dir.strip():
        return "❌ Please select an input directory"

    if not output_dir or not output_dir.strip():
        return "❌ Please select an output directory"

    input_dir = input_dir.strip()
    output_dir = output_dir.strip()

    if not os.path.exists(input_dir):
        return f"❌ Input directory not found: {input_dir}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find audio files
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    audio_files = []
    for f in os.listdir(input_dir):
        if Path(f).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(input_dir, f))

    if not audio_files:
        return f"❌ No audio files found in {input_dir}"

    progress(0.1, desc=f"Found {len(audio_files)} audio files...")

    try:
        from pydub import AudioSegment
    except ImportError:
        return "❌ pydub not installed. Run: pip install pydub"

    total_segments = 0
    processed_files = 0
    segment_ms = segment_duration * 1000  # Convert to milliseconds

    for i, audio_path in enumerate(audio_files):
        try:
            progress((i + 1) / len(audio_files), desc=f"Processing {Path(audio_path).name}...")

            # Load audio
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)

            # Get base filename without extension
            base_name = Path(audio_path).stem

            # Split into segments
            segment_idx = 0
            start_ms = 0

            while start_ms < duration_ms:
                end_ms = min(start_ms + segment_ms, duration_ms)

                # Only save if segment is at least 10 seconds
                if (end_ms - start_ms) >= 10000:
                    segment = audio[start_ms:end_ms]

                    # Save segment
                    segment_filename = f"{base_name}_seg{segment_idx:02d}.wav"
                    segment_path = os.path.join(output_dir, segment_filename)
                    segment.export(segment_path, format="wav")

                    total_segments += 1
                    segment_idx += 1

                start_ms = end_ms

            processed_files += 1

        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            continue

    progress(1.0, desc="Done!")

    return f"✅ Split {processed_files} files into {total_segments} segments ({segment_duration}s each) → {output_dir}"


# ============== Dataset Builder Functions ==============

def create_dataset_builder():
    from acestep.training.dataset_builder import DatasetBuilder
    return DatasetBuilder()


def scan_audio_directory(
    audio_dir: str,
    dataset_name: str,
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    genre_ratio: int,
    builder_state,
    progress=gr.Progress(),
) -> Tuple[Any, str, Any, Any, str]:
    """Scan directory for audio files."""
    global workflow_state

    if not audio_dir or not audio_dir.strip():
        return [], "❌ Please select an audio directory", gr.Slider(maximum=0, value=0), builder_state, get_status_html()

    progress(0.1, desc="Scanning directory...")

    from acestep.training.dataset_builder import DatasetBuilder
    builder = DatasetBuilder()

    builder.metadata.name = dataset_name
    builder.metadata.custom_tag = custom_tag
    builder.metadata.tag_position = tag_position
    builder.metadata.all_instrumental = all_instrumental
    builder.metadata.genre_ratio = int(genre_ratio)

    progress(0.3, desc="Loading audio files...")
    samples, status = builder.scan_directory(audio_dir.strip())

    if not samples:
        return [], status, gr.Slider(maximum=0, value=0), builder, get_status_html()

    progress(0.6, desc="Processing samples...")
    builder.set_all_instrumental(all_instrumental)
    if custom_tag:
        builder.set_custom_tag(custom_tag, tag_position)

    # Check for pre-generated captions from ACE-Step Captioner
    progress(0.7, desc="Checking for pre-generated captions...")
    captions_loaded = 0
    audio_dir_path = Path(audio_dir.strip())

    for sample in builder.samples:
        # Look for JSON file with same name as audio file
        sample_stem = Path(sample.audio_path).stem
        caption_json = audio_dir_path / f"{sample_stem}.json"

        if caption_json.exists():
            try:
                with open(caption_json, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                    if "caption" in caption_data:
                        sample.caption = caption_data["caption"]
                        sample.labeled = True
                        captions_loaded += 1
                    if "lyrics" in caption_data:
                        sample.lyrics = caption_data["lyrics"]
                    if "bpm" in caption_data and caption_data["bpm"]:
                        sample.bpm = int(caption_data["bpm"])
                    if "keyscale" in caption_data and caption_data["keyscale"]:
                        sample.keyscale = caption_data["keyscale"]
                    if "timesignature" in caption_data and caption_data["timesignature"]:
                        sample.timesignature = str(caption_data["timesignature"])
                    if "duration" in caption_data and caption_data["duration"]:
                        sample.duration = int(caption_data["duration"])
                    if "genre" in caption_data and caption_data["genre"]:
                        sample.genre = caption_data["genre"]
                    if "language" in caption_data and caption_data["language"]:
                        sample.language = caption_data["language"]
                    if "is_instrumental" in caption_data:
                        sample.is_instrumental = bool(caption_data["is_instrumental"])
            except Exception as e:
                logger.warning(f"Failed to load caption from {caption_json}: {e}")

    progress(0.9, desc="Building table...")
    table_data = builder.get_samples_dataframe_data()
    slider_max = max(0, len(samples) - 1)

    workflow_state["dataset_loaded"] = True
    workflow_state["dataset_labeled"] = captions_loaded > 0

    # Update status with caption info
    if captions_loaded > 0:
        status = f"{status}\n🏷️ Loaded {captions_loaded} pre-generated captions from JSON files"

    progress(1.0, desc="Done!")
    return table_data, status, gr.Slider(maximum=slider_max, value=0), builder, get_status_html()


def load_existing_dataset(dataset_path: str, builder_state, progress=gr.Progress()):
    """Load existing dataset JSON."""
    global workflow_state

    if not dataset_path or not dataset_path.strip():
        return "❌ Please select a dataset file", [], gr.Slider(maximum=0, value=0), builder_state, get_status_html()

    progress(0.2, desc="Loading dataset...")

    from acestep.training.dataset_builder import DatasetBuilder
    builder = DatasetBuilder()
    samples, status = builder.load_dataset(dataset_path.strip())

    if not samples:
        return status, [], gr.Slider(maximum=0, value=0), builder, get_status_html()

    progress(0.8, desc="Building table...")
    table_data = builder.get_samples_dataframe_data()
    slider_max = max(0, len(samples) - 1)

    labeled_count = builder.get_labeled_count()
    workflow_state["dataset_loaded"] = True
    workflow_state["dataset_labeled"] = labeled_count > 0

    info = f"✅ Loaded: {builder.metadata.name} | {len(samples)} samples ({labeled_count} labeled)"

    progress(1.0, desc="Done!")
    return info, table_data, gr.Slider(maximum=slider_max, value=0), builder, get_status_html()


def auto_label_samples(skip_metas: bool, only_unlabeled: bool, builder_state, progress=gr.Progress()):
    """Auto-label samples using AI."""
    global workflow_state

    if builder_state is None or not builder_state.samples:
        return [], "❌ Load a dataset first", builder_state, get_status_html()

    if dit_handler is None or dit_handler.model is None:
        return builder_state.get_samples_dataframe_data(), "❌ Initialize service first (Step 1)", builder_state, get_status_html()

    if llm_handler is None or not llm_handler.llm_initialized:
        return builder_state.get_samples_dataframe_data(), "❌ LLM not available. Use CSV metadata or manual labeling.", builder_state, get_status_html()

    def progress_callback(msg):
        progress(0.5, desc=msg)

    progress(0.1, desc="Starting auto-labeling...")

    samples, status = builder_state.label_all_samples(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        format_lyrics=False,
        transcribe_lyrics=False,
        skip_metas=skip_metas,
        only_unlabeled=only_unlabeled,
        progress_callback=progress_callback,
    )

    workflow_state["dataset_labeled"] = builder_state.get_labeled_count() > 0

    progress(1.0, desc="Done!")
    return builder_state.get_samples_dataframe_data(), status, builder_state, get_status_html()


def get_sample_preview(sample_idx: int, builder_state):
    """Get preview for selected sample."""
    empty = (None, "", "", "", "Use Global Ratio", "", None, "", "", 0.0, "instrumental", True)

    if builder_state is None or not builder_state.samples:
        return empty

    idx = int(sample_idx)
    if idx < 0 or idx >= len(builder_state.samples):
        return empty

    sample = builder_state.samples[idx]

    override_choice = "Use Global Ratio"
    if sample.prompt_override == "genre":
        override_choice = "Genre"
    elif sample.prompt_override == "caption":
        override_choice = "Caption"

    return (
        sample.audio_path,
        sample.filename,
        sample.caption,
        sample.genre,
        override_choice,
        sample.lyrics,
        sample.bpm,
        sample.keyscale,
        sample.timesignature,
        sample.duration,
        sample.language,
        sample.is_instrumental,
    )


def save_sample_edit(
    sample_idx, caption, genre, prompt_override, lyrics, bpm, keyscale, timesig, language, is_instrumental, builder_state
):
    """Save edits to sample."""
    if builder_state is None:
        return [], "❌ No dataset loaded", builder_state

    override_value = None
    if prompt_override == "Genre":
        override_value = "genre"
    elif prompt_override == "Caption":
        override_value = "caption"

    sample, status = builder_state.update_sample(
        int(sample_idx),
        caption=caption,
        genre=genre,
        prompt_override=override_value,
        lyrics=lyrics if not is_instrumental else "[Instrumental]",
        bpm=int(bpm) if bpm else None,
        keyscale=keyscale,
        timesignature=timesig,
        language="unknown" if is_instrumental else language,
        is_instrumental=is_instrumental,
        labeled=True,
    )

    return builder_state.get_samples_dataframe_data(), status, builder_state


def save_dataset(save_path: str, dataset_name: str, custom_tag: str, tag_position: str, builder_state) -> str:
    """Save dataset to JSON. Re-applies tag from UI to ensure it's always saved."""
    if builder_state is None or not builder_state.samples:
        return "❌ No dataset to save"
    if not save_path:
        return "❌ Please specify a save path"

    # Failsafe: re-apply tag from UI fields
    tag = custom_tag.strip() if custom_tag else ""
    if not tag:
        return "❌ Activation Tag is empty! Set a tag before saving."

    builder_state.metadata.custom_tag = tag
    builder_state.metadata.tag_position = tag_position
    for sample in builder_state.samples:
        sample.custom_tag = tag

    # Create directory if needed
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    return builder_state.save_dataset(save_path.strip(), dataset_name)


# ============== Preprocessing Functions ==============

def preprocess_dataset(
    output_dir: str, max_duration: float,
    custom_tag: str, tag_position: str,
    builder_state, progress=gr.Progress(),
) -> Tuple[str, str]:
    """Preprocess dataset to tensors.

    Reads custom_tag and tag_position directly from the UI fields
    to guarantee they are always up-to-date (not relying on stale builder_state).
    """
    global workflow_state

    if builder_state is None or not builder_state.samples:
        return "❌ Load a dataset first", get_status_html()

    labeled_count = builder_state.get_labeled_count()
    if labeled_count == 0:
        return "❌ No labeled samples. Run auto-labeling first.", get_status_html()

    if dit_handler is None or dit_handler.model is None:
        return "❌ Initialize service first", get_status_html()

    if not output_dir:
        return "❌ Please specify output directory", get_status_html()

    # ── FAILSAFE: activation tag is mandatory ──
    tag = custom_tag.strip() if custom_tag else ""
    if not tag:
        return "❌ Activation Tag is empty! Set an activation tag in the Dataset tab before preprocessing.", get_status_html()

    # Always re-apply the tag from UI fields onto builder + all samples
    # This ensures the tag is never lost even if gr.State was stale
    builder_state.metadata.custom_tag = tag
    builder_state.metadata.tag_position = tag_position
    for sample in builder_state.samples:
        sample.custom_tag = tag

    logger.info(f"[Preprocess] Activation tag='{tag}', position='{tag_position}', "
                f"samples={len(builder_state.samples)}, "
                f"genre_ratio={builder_state.metadata.genre_ratio}")
    if builder_state.samples:
        s = builder_state.samples[0]
        prompt_preview = s.get_training_prompt(tag_position, use_genre=False)
        logger.info(f"[Preprocess] Sample[0] prompt preview: '{prompt_preview[:120]}...'")

    # Create output directory
    os.makedirs(output_dir.strip(), exist_ok=True)

    def progress_callback(msg):
        progress(0.5, desc=msg)

    progress(0.1, desc="Starting preprocessing...")

    output_paths, status = builder_state.preprocess_to_tensors(
        dit_handler=dit_handler,
        output_dir=output_dir.strip(),
        max_duration=max_duration,
        progress_callback=progress_callback,
    )

    workflow_state["tensors_ready"] = len(output_paths) > 0

    progress(1.0, desc="Done!")
    return status, get_status_html()


# ============== Training Functions ==============

def _recommend_epochs(num_samples: int) -> tuple:
    """Recommend max_epochs and save_every based on sample count.

    Based on community feedback: fewer samples need more epochs,
    but too many cause overfitting.

    Returns:
        (max_epochs, save_every_n_epochs)
    """
    if num_samples <= 0:
        return 1000, 200
    if num_samples <= 3:
        return 1500, 200
    if num_samples <= 6:
        return 1000, 200
    if num_samples <= 10:
        return 700, 100
    if num_samples <= 20:
        return 500, 100
    if num_samples <= 50:
        return 300, 50
    return 200, 50


def load_tensor_dataset(tensor_dir: str):
    """Load tensor dataset info and recommend epochs based on sample count.

    Returns:
        (info_text, recommended_max_epochs, recommended_save_every)
    """
    if not tensor_dir:
        return "❌ Please specify tensor directory", gr.skip(), gr.skip()

    tensor_dir = tensor_dir.strip()
    if not os.path.exists(tensor_dir):
        return f"❌ Directory not found: {tensor_dir}", gr.skip(), gr.skip()

    num_samples = 0

    manifest_path = os.path.join(tensor_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            num_samples = manifest.get("num_samples", 0)
            metadata = manifest.get("metadata", {})
            name = metadata.get("name", "Unknown")
            custom_tag = metadata.get("custom_tag", "")
            rec_epochs, rec_save = _recommend_epochs(num_samples)
            info = (
                f"✅ Dataset: {name}\n"
                f"📊 {num_samples} tensors\n"
                f"🏷️ Tag: {custom_tag or '(none)'}\n"
                f"💡 Recommended: ~{rec_epochs} epochs, save every {rec_save}"
            )
            return info, rec_epochs, rec_save
        except:
            pass

    pt_files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
    if not pt_files:
        return f"❌ No .pt files found in {tensor_dir}", gr.skip(), gr.skip()

    num_samples = len(pt_files)
    rec_epochs, rec_save = _recommend_epochs(num_samples)
    info = (
        f"✅ Found {num_samples} tensor files\n"
        f"💡 Recommended: ~{rec_epochs} epochs, save every {rec_save}"
    )
    return info, rec_epochs, rec_save


def _format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def start_training(
    tensor_dir, lora_rank, lora_alpha, lora_dropout, learning_rate, max_epochs,
    batch_size, gradient_accumulation, save_every_n_epochs, shift, seed, output_dir,
    early_stop_enabled, early_stop_patience_val,
    auto_save_best_after_val,
    max_latent_len, torch_compile_flag,
    model_type_val, guidance_scale_val, cfg_dropout_val, num_inference_steps_val,
    optimizer_type_val, scheduler_type_val, attention_type_val,
    gradient_checkpointing_flag, encoder_offloading_flag,
    resume_enabled, resume_dir, resume_checkpoint,
    training_state,
    progress=gr.Progress(),
):
    """Start LoRA training."""
    global workflow_state
    import re

    if not tensor_dir:
        yield "❌ Please specify tensor directory", "", None, training_state, get_status_html()
        return

    if not os.path.exists(tensor_dir.strip()):
        yield f"❌ Directory not found: {tensor_dir}", "", None, training_state, get_status_html()
        return

    if dit_handler is None or dit_handler.model is None:
        yield "❌ Initialize service first", "", None, training_state, get_status_html()
        return

    try:
        import torch
        torch.set_float32_matmul_precision('medium')
        from lightning.fabric import Fabric
        from peft import get_peft_model, LoraConfig
    except ImportError as e:
        yield f"❌ Missing packages: {e}\nInstall: pip install peft lightning", "", None, training_state, get_status_html()
        return

    training_state["is_training"] = True
    training_state["should_stop"] = False

    try:
        from acestep.training.trainer import LoRATrainer
        from acestep.training.configs import LoRAConfig as LoRAConfigClass, TrainingConfig

        lora_config = LoRAConfigClass(r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

        # Determine model type from UI or global detection
        resolved_model_type = model_type_val if model_type_val else current_model_type
        is_base = resolved_model_type == "base"

        training_config = TrainingConfig(
            model_type=resolved_model_type,
            shift=shift,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            max_epochs=max_epochs,
            save_every_n_epochs=save_every_n_epochs,
            seed=seed,
            output_dir=output_dir,
            early_stop_patience=int(early_stop_patience_val) if early_stop_enabled else 0,
            auto_save_best_after=int(auto_save_best_after_val),
            max_latent_length=int(max_latent_len),
            torch_compile=bool(torch_compile_flag),
            # Base model specific params
            guidance_scale=float(guidance_scale_val) if is_base else 0.0,
            cfg_dropout_prob=float(cfg_dropout_val) if is_base else 0.1,
            num_inference_steps=int(num_inference_steps_val) if is_base else 8,
            # New features
            optimizer_type=str(optimizer_type_val) if optimizer_type_val else "adamw",
            scheduler_type=str(scheduler_type_val) if scheduler_type_val else "cosine",
            attention_type=str(attention_type_val) if attention_type_val else "both",
            gradient_checkpointing=bool(gradient_checkpointing_flag),
            encoder_offloading=bool(encoder_offloading_flag),
        )

        log_lines = []
        loss_data = pd.DataFrame({"step": [0], "loss": [0.0]})
        start_time = time.time()

        type_label = "⚡ Turbo" if not is_base else f"🎯 Base (CFG={guidance_scale_val}, steps={num_inference_steps_val})"
        opt_label = optimizer_type_val or "adamw"
        yield f"🚀 Starting {type_label} training ({opt_label})...", "", loss_data, training_state, get_status_html()

        trainer = LoRATrainer(
            dit_handler=dit_handler,
            lora_config=lora_config,
            training_config=training_config,
        )

        step_list, loss_list = [], []

        # Resolve resume path
        resume_path = None
        if resume_enabled and resume_dir and resume_checkpoint:
            resume_path = os.path.join(resume_dir.strip(), resume_checkpoint)
            if not os.path.exists(resume_path):
                yield f"⚠️ Resume checkpoint not found: {resume_path}, starting fresh", "", loss_data, training_state, get_status_html()
                resume_path = None
            else:
                yield f"🔄 Will resume from {resume_checkpoint}...", "", loss_data, training_state, get_status_html()

        for step, loss, status in trainer.train_from_preprocessed(tensor_dir.strip(), training_state, resume_from=resume_path):
            elapsed = time.time() - start_time
            time_info = f"⏱️ {_format_duration(elapsed)}"

            match = re.search(r"Epoch\s+(\d+)/(\d+)", str(status))
            if match:
                current_ep, total_ep = int(match.group(1)), int(match.group(2))
                if current_ep > 0:
                    eta = (elapsed / current_ep) * (total_ep - current_ep)
                    time_info += f" | ETA: ~{_format_duration(eta)}"

            log_lines.append(status)
            if len(log_lines) > 12:
                log_lines = log_lines[-12:]

            if step > 0 and loss is not None and loss == loss:
                step_list.append(step)
                loss_list.append(float(loss))
                loss_data = pd.DataFrame({"step": step_list, "loss": loss_list})

            yield f"{status}\n{time_info}", "\n".join(log_lines), loss_data, training_state, get_status_html()

            if training_state.get("should_stop", False):
                log_lines.append("⏹️ Stopped by user")
                yield f"⏹️ Stopped | {time_info}", "\n".join(log_lines[-12:]), loss_data, training_state, get_status_html()
                break

        total_time = time.time() - start_time
        training_state["is_training"] = False
        workflow_state["training_complete"] = True

        completion_msg = f"✅ Training complete! Time: {_format_duration(total_time)}"
        log_lines.append(completion_msg)

        yield completion_msg, "\n".join(log_lines[-12:]), loss_data, training_state, get_status_html()

    except Exception as e:
        logger.exception("Training error")
        training_state["is_training"] = False
        yield f"❌ Error: {str(e)}", str(e), pd.DataFrame({"step": [], "loss": []}), training_state, get_status_html()


# ==================== GPU TRAINING PRESETS ====================
# Each preset is a dict of UI values optimized for a specific GPU tier.
# Keys must match the Gradio component variable names.
GPU_PRESETS = {
    "Custom": None,  # No changes — user's current settings
    "RTX 4090 / 5090 (24GB+)": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        "max_epochs": 800,
        "batch_size": 3,
        "gradient_accumulation": 1,
        "optimizer_type": "prodigy",
        "scheduler_type": "cosine",
        "attention_type": "both",
        "gradient_checkpointing": False,
        "encoder_offloading": False,
        "torch_compile": False,
        "early_stop_enabled": True,
        "early_stop_patience": 80,
        "auto_save_best_after": 200,
        "save_every_n_epochs": 50,
        "max_latent_length": 1500,
    },
    "RTX 3090 / 4080 (16-24GB)": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        "max_epochs": 800,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "optimizer_type": "prodigy",
        "scheduler_type": "cosine",
        "attention_type": "both",
        "gradient_checkpointing": False,
        "encoder_offloading": False,
        "torch_compile": False,
        "early_stop_enabled": True,
        "early_stop_patience": 80,
        "auto_save_best_after": 200,
        "save_every_n_epochs": 50,
        "max_latent_length": 1500,
    },
    "RTX 3080 / 4070 (10-12GB)": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        "max_epochs": 1000,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "optimizer_type": "adamw8bit",
        "scheduler_type": "cosine",
        "attention_type": "both",
        "gradient_checkpointing": True,
        "encoder_offloading": True,
        "torch_compile": False,
        "early_stop_enabled": True,
        "early_stop_patience": 80,
        "auto_save_best_after": 200,
        "save_every_n_epochs": 50,
        "max_latent_length": 1500,
    },
    "RTX 3060 / 4060 (8GB)": {
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        "max_epochs": 1000,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "optimizer_type": "adafactor",
        "scheduler_type": "constant_with_warmup",
        "attention_type": "self",
        "gradient_checkpointing": True,
        "encoder_offloading": True,
        "torch_compile": False,
        "early_stop_enabled": True,
        "early_stop_patience": 80,
        "auto_save_best_after": 200,
        "save_every_n_epochs": 50,
        "max_latent_length": 1000,
    },
}


def apply_gpu_preset(preset_name):
    """Apply a GPU preset to all training parameters.

    Returns a tuple of gr.update() calls matching the order of output components.
    """
    preset = GPU_PRESETS.get(preset_name)
    if preset is None:
        # "Custom" — no changes
        return (
            gr.update(),  # lora_rank
            gr.update(),  # lora_alpha
            gr.update(),  # lora_dropout
            gr.update(),  # learning_rate
            gr.update(),  # max_epochs
            gr.update(),  # batch_size
            gr.update(),  # gradient_accumulation
            gr.update(),  # optimizer_type
            gr.update(),  # scheduler_type
            gr.update(),  # attention_type
            gr.update(),  # gradient_checkpointing
            gr.update(),  # encoder_offloading
            gr.update(),  # torch_compile
            gr.update(),  # early_stop_enabled
            gr.update(),  # early_stop_patience
            gr.update(),  # auto_save_best_after
            gr.update(),  # save_every_n_epochs
            gr.update(),  # max_latent_length
        )

    return (
        gr.update(value=preset["lora_rank"]),
        gr.update(value=preset["lora_alpha"]),
        gr.update(value=preset["lora_dropout"]),
        gr.update(value=preset["learning_rate"]),
        gr.update(value=preset["max_epochs"]),
        gr.update(value=preset["batch_size"]),
        gr.update(value=preset["gradient_accumulation"]),
        gr.update(value=preset["optimizer_type"]),
        gr.update(value=preset["scheduler_type"]),
        gr.update(value=preset["attention_type"]),
        gr.update(value=preset["gradient_checkpointing"]),
        gr.update(value=preset["encoder_offloading"]),
        gr.update(value=preset["torch_compile"]),
        gr.update(value=preset["early_stop_enabled"]),
        gr.update(value=preset["early_stop_patience"]),
        gr.update(value=preset["auto_save_best_after"]),
        gr.update(value=preset["save_every_n_epochs"]),
        gr.update(value=preset["max_latent_length"]),
    )


def run_gradient_estimation(tensor_dir, max_batches, granularity, top_k, progress=gr.Progress()):
    """Run gradient sensitivity estimation on the dataset."""
    import pandas as pd

    empty_df = pd.DataFrame(columns=["Rank", "Module", "Score", "Raw Norm"])

    if dit_handler is None or dit_handler.model is None:
        yield "❌ Initialize service first (Step ①)", empty_df
        return

    if not tensor_dir or not os.path.exists(tensor_dir.strip()):
        yield f"❌ Tensor directory not found: {tensor_dir}", empty_df
        return

    try:
        from acestep.training.estimator import estimate_gradient_sensitivity
        from acestep.training.configs import TrainingConfig

        training_config = TrainingConfig(
            model_type=current_model_type or "turbo",
            batch_size=1,
            max_latent_length=1500,
        )

        for pct, status, results in estimate_gradient_sensitivity(
            dit_handler=dit_handler,
            tensor_dir=tensor_dir.strip(),
            training_config=training_config,
            max_batches=int(max_batches),
            granularity=granularity,
            top_k=int(top_k),
        ):
            if results is not None:
                # Final result
                rows = []
                for i, r in enumerate(results):
                    rows.append([i + 1, r["module"], f"{r['score']:.4f}", f"{r['raw_norm']:.6f}"])
                df = pd.DataFrame(rows, columns=["Rank", "Module", "Score", "Raw Norm"])
                yield status, df
            else:
                yield status, empty_df

    except Exception as e:
        logger.exception("Estimation error")
        yield f"❌ Error: {str(e)}", empty_df


def stop_training(training_state):
    """Stop training."""
    if not training_state.get("is_training", False):
        return "⚠️ No training in progress", training_state
    training_state["should_stop"] = True
    return "⏹️ Stopping...", training_state


def export_lora(export_path: str, output_dir: str) -> str:
    """Export trained LoRA."""
    if not export_path:
        return "❌ Please specify export path"

    import shutil

    final_dir = os.path.join(output_dir, "final")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if os.path.exists(final_dir):
        source_path = final_dir
    elif os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if not checkpoints:
            return "❌ No checkpoints found"
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        source_path = os.path.join(checkpoint_dir, checkpoints[-1])
    else:
        return f"❌ No trained model in {output_dir}"

    try:
        export_path = export_path.strip()
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else ".", exist_ok=True)
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        shutil.copytree(source_path, export_path)
        return f"✅ LoRA exported to {export_path}"
    except Exception as e:
        return f"❌ Export failed: {str(e)}"


# ============== LoRA Merge ==============

def list_lora_checkpoints(output_dir: str) -> List[str]:
    """List available LoRA checkpoints in an output directory."""
    checkpoints = []

    if not output_dir or not os.path.exists(output_dir):
        return checkpoints

    # Check for /final
    final_dir = os.path.join(output_dir, "final", "adapter")
    if os.path.exists(final_dir) and os.path.exists(os.path.join(final_dir, "adapter_config.json")):
        checkpoints.append(f"final")

    # Check numbered epoch checkpoints
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        for d in sorted(os.listdir(ckpt_dir)):
            if d.startswith("epoch_"):
                adapter_path = os.path.join(ckpt_dir, d, "adapter")
                if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                    checkpoints.append(f"checkpoints/{d}")

    return checkpoints


def scan_lora_checkpoints(lora_dir: str):
    """Scan for available LoRA checkpoints and return dropdown update."""
    checkpoints = list_lora_checkpoints(lora_dir)
    if not checkpoints:
        return gr.Dropdown(choices=[], value=None), "❌ No LoRA checkpoints found"

    # Default to "final" if available, otherwise last epoch
    default = checkpoints[0] if "final" in checkpoints[0] else checkpoints[-1]
    return gr.Dropdown(choices=checkpoints, value=default), f"✅ Found {len(checkpoints)} checkpoint(s)"


def merge_lora_to_safetensors(
    checkpoint_name: str,
    lora_dir: str,
    lora_checkpoint: str,
    merge_output_dir: str,
    progress=gr.Progress()
) -> str:
    """Merge LoRA adapter weights into the base model and save as safetensors.

    This creates a standalone merged model that can be used without PEFT,
    compatible with ComfyUI and other tools that don't support LoRA adapters.

    Args:
        checkpoint_name: Name of the base model checkpoint (e.g. 'acestep-v15-turbo')
        lora_dir: Directory containing training output (with checkpoints/ or final/)
        lora_checkpoint: Which checkpoint to use (e.g. 'final' or 'checkpoints/epoch_50')
        merge_output_dir: Where to save the merged model

    Returns:
        Status message
    """
    import shutil
    import gc

    if not checkpoint_name:
        return "❌ Please select a base model checkpoint"
    if not lora_checkpoint:
        return "❌ Please select a LoRA checkpoint"
    if not merge_output_dir:
        return "❌ Please specify an output directory"

    # Resolve paths
    base_model_path = str(get_checkpoints_dir() / checkpoint_name)
    if not os.path.exists(base_model_path):
        return f"❌ Base model not found: {base_model_path}"

    # Find the adapter directory
    adapter_path = os.path.join(lora_dir, lora_checkpoint, "adapter")
    if not os.path.exists(adapter_path):
        # Maybe it's the directory itself (no "adapter" subfolder)
        adapter_path = os.path.join(lora_dir, lora_checkpoint)
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        return f"❌ adapter_config.json not found in {adapter_path}"

    try:
        progress(0.05, desc="Importing libraries...")
        import torch
        from transformers import AutoModel
        from peft import PeftModel

        # Step 1: Load base model
        progress(0.1, desc="Loading base model (this takes a moment)...")
        logger.info(f"[merge] Loading base model from {base_model_path}")

        base_model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            dtype="bfloat16"
        )
        base_model = base_model.to("cpu").to(torch.bfloat16)
        base_model.eval()

        progress(0.4, desc="Loading LoRA adapter...")
        logger.info(f"[merge] Loading LoRA adapter from {adapter_path}")

        # Step 2: Apply LoRA adapter to the decoder
        base_model.decoder = PeftModel.from_pretrained(
            base_model.decoder,
            adapter_path,
        )

        progress(0.6, desc="Merging LoRA weights into base model...")
        logger.info("[merge] Merging LoRA weights...")

        # Step 3: Merge and unload — permanently integrates LoRA into weights
        base_model.decoder = base_model.decoder.merge_and_unload()

        progress(0.75, desc="Saving merged model as safetensors...")
        logger.info(f"[merge] Saving merged model to {merge_output_dir}")

        # Step 4: Save merged model
        os.makedirs(merge_output_dir, exist_ok=True)
        base_model.save_pretrained(merge_output_dir, safe_serialization=True)

        # Step 5: Copy silence_latent.pt (needed for inference)
        silence_src = os.path.join(base_model_path, "silence_latent.pt")
        silence_dst = os.path.join(merge_output_dir, "silence_latent.pt")
        if os.path.exists(silence_src) and not os.path.exists(silence_dst):
            shutil.copy2(silence_src, silence_dst)
            logger.info("[merge] Copied silence_latent.pt")

        progress(0.95, desc="Cleaning up...")

        # Free memory
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Verify output
        merged_weights = os.path.join(merge_output_dir, "model.safetensors")
        if os.path.exists(merged_weights):
            size_gb = os.path.getsize(merged_weights) / (1024 ** 3)
            progress(1.0, desc="Done!")
            return (
                f"✅ Merged model saved to {merge_output_dir}\n"
                f"   model.safetensors: {size_gb:.1f} GB\n"
                f"   Base: {checkpoint_name} + LoRA: {lora_checkpoint}\n"
                f"   Ready for ComfyUI or direct inference!"
            )
        else:
            # Check for sharded output
            shards = [f for f in os.listdir(merge_output_dir) if f.startswith("model") and f.endswith(".safetensors")]
            if shards:
                total_size = sum(os.path.getsize(os.path.join(merge_output_dir, f)) for f in shards) / (1024 ** 3)
                progress(1.0, desc="Done!")
                return (
                    f"✅ Merged model saved to {merge_output_dir}\n"
                    f"   {len(shards)} shard(s), total: {total_size:.1f} GB\n"
                    f"   Base: {checkpoint_name} + LoRA: {lora_checkpoint}\n"
                    f"   Ready for ComfyUI or direct inference!"
                )
            return f"⚠️ Model saved but safetensors not found in {merge_output_dir}"

    except Exception as e:
        logger.exception("Failed to merge LoRA")
        # Cleanup on error
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        return f"❌ Merge failed: {str(e)}"


# ============== Build Gradio Interface ==============

CSS = """
/* Status Bar */
.status-bar {
    display: flex;
    justify-content: space-between;
    padding: 12px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #0f3460;
}
.status-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.3s ease;
}
.step-done {
    background: linear-gradient(135deg, #00d9a5 0%, #00b894 100%);
    color: #fff;
    box-shadow: 0 2px 8px rgba(0, 217, 165, 0.3);
}
.step-pending {
    background: rgba(255,255,255,0.1);
    color: #888;
}
.step-icon { font-size: 16px; }
.step-name { font-size: 13px; }

/* Section Cards */
.section-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #1a1a35 100%);
    border: 1px solid #333366;
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
}
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Compact inputs */
.compact-row { gap: 10px !important; }

/* Action buttons */
.primary-action {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
}
.primary-action:hover {
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

/* Info box */
.info-box {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #a0a0c0;
}

/* Quick tip */
.quick-tip {
    background: rgba(0, 217, 165, 0.1);
    border-left: 3px solid #00d9a5;
    padding: 10px 15px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #a0c0b0;
}

/* Header */
.app-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid #0f3460;
}
.app-header h1 {
    font-size: 28px;
    margin: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p { color: #888; margin: 8px 0 0 0; font-size: 14px; }
"""

def create_ui():
    """Create the Gradio interface with improved UX."""

    with gr.Blocks(title="ACE-Step LoRA Training", css=CSS) as demo:

        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🎵 ACE-Step LoRA Training</h1>
            <p>Train custom style adapters • Based on the Master Guide by Moonspell & AI Brother</p>
        </div>
        """)

        # Status Bar
        status_bar = gr.HTML(get_status_html())

        # State
        dataset_builder_state = gr.State(None)
        training_state = gr.State({"is_training": False, "should_stop": False})

        # ==================== STEP 1: SERVICE ====================
        with gr.Tab("① Service", id="tab_service"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">🔧 Model Configuration</div>')

                    with gr.Row(elem_classes="compact-row"):
                        checkpoint_dropdown = gr.Dropdown(
                            choices=list_available_checkpoints(),
                            label="Model Checkpoint",
                            info="Select ACE-Step model (turbo = fast, base = quality)",
                            scale=4,
                        )
                        refresh_btn = gr.Button("🔄", scale=0, min_width=50)

                    with gr.Row():
                        init_btn = gr.Button("🚀 Initialize Service", variant="primary", scale=2, elem_classes="primary-action")
                        unload_btn = gr.Button("🗑️ Unload", variant="stop", scale=1)

                    service_status = gr.Textbox(label="Status", interactive=False, lines=1)

                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="quick-tip">
                        <strong>💡 Quick Tip</strong><br>
                        • <b>turbo</b>: Train with Shift=3.0<br>
                        • <b>base</b>: Train with Shift=1.0<br>
                        Match training Shift to inference!
                    </div>
                    """)

        # ==================== STEP 2: DATASET ====================
        with gr.Tab("② Dataset", id="tab_dataset"):

            # Audio Splitter Section (collapsible)
            with gr.Accordion("✂️ Split Long Audio Files (Recommended for faster training)", open=False):
                gr.HTML("""
                <div class="info-box">
                    <strong>⚡ Speed Tip:</strong> Training is much faster with shorter audio segments (30-60s).
                    Long files (3+ minutes) can make training 20x slower!
                    Use this tool to split your audio files before scanning.
                </div>
                """)
                with gr.Row(elem_classes="compact-row"):
                    split_input_dir = gr.Textbox(
                        label="Input Folder (original audio)",
                        placeholder="Select folder with long audio files...",
                        scale=3,
                    )
                    split_input_picker = gr.Button("📁", scale=0, min_width=45)

                with gr.Row(elem_classes="compact-row"):
                    split_output_dir = gr.Textbox(
                        label="Output Folder (split segments)",
                        placeholder="Select folder to save segments...",
                        scale=3,
                    )
                    split_output_picker = gr.Button("📁", scale=0, min_width=45)

                with gr.Row(elem_classes="compact-row"):
                    split_duration = gr.Radio(
                        choices=[30, 60],
                        value=30,
                        label="Segment Duration (seconds)",
                        info="30s = fastest training, 60s = more context",
                    )
                    split_btn = gr.Button("✂️ Split Audio Files", variant="primary", scale=1)

                split_status = gr.Textbox(label="Split Status", interactive=False)

            with gr.Row():
                # Left: Load/Scan
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📂 Load Audio Files</div>')

                    with gr.Accordion("🔍 Scan New Directory", open=True):
                        with gr.Row(elem_classes="compact-row"):
                            audio_directory = gr.Textbox(
                                label="Audio Folder",
                                placeholder="Select folder with audio files...",
                                scale=4,
                            )
                            audio_dir_picker = gr.Button("📁", scale=0, min_width=45)
                        scan_btn = gr.Button("🔍 Scan", variant="primary", size="sm")

                    with gr.Accordion("📄 Load Existing Dataset", open=False):
                        with gr.Row(elem_classes="compact-row"):
                            load_dataset_path = gr.Textbox(
                                label="Dataset JSON",
                                placeholder="Select .json file...",
                                scale=4,
                            )
                            load_dataset_picker = gr.Button("📄", scale=0, min_width=45)
                        load_dataset_btn = gr.Button("📂 Load", variant="secondary", size="sm")

                    scan_status = gr.Textbox(label="Status", interactive=False, lines=2)

                # Right: Settings
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">⚙️ Dataset Settings</div>')

                    dataset_name = gr.Textbox(label="Dataset Name", value="my_lora_dataset")

                    with gr.Row(elem_classes="compact-row"):
                        custom_tag = gr.Textbox(
                            label="Activation Tag",
                            placeholder="e.g., ZX_MyStyle",
                            info="Unique trigger word for your LoRA",
                        )
                        tag_position = gr.Dropdown(
                            choices=["prepend", "append", "replace"],
                            value="replace",
                            label="Position",
                        )

                    with gr.Row(elem_classes="compact-row"):
                        all_instrumental = gr.Checkbox(label="All Instrumental", value=True)
                        genre_ratio = gr.Slider(0, 100, 30, step=10, label="Genre Ratio %", info="30% recommended")

            # Audio Files Table
            gr.HTML('<div class="section-title" style="margin-top:20px">📊 Audio Files</div>')
            audio_files_table = gr.Dataframe(
                headers=["#", "Filename", "Duration", "Lyrics", "Labeled", "BPM", "Key", "Caption"],
                datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                max_height=300,
            )

            # Auto-Label Section
            with gr.Accordion("🤖 Auto-Label with AI", open=True):
                gr.HTML('<div class="info-box">Uses AI to generate captions, genre tags, BPM, key, and time signature for each audio file.</div>')
                with gr.Row():
                    skip_metas = gr.Checkbox(label="Skip BPM/Key (use CSV)", value=False)
                    only_unlabeled = gr.Checkbox(label="Only unlabeled", value=False)
                    auto_label_btn = gr.Button("🏷️ Auto-Label All", variant="primary", elem_classes="primary-action")
                label_progress = gr.Textbox(label="Progress", interactive=False)

            # Preview & Edit
            with gr.Accordion("👀 Preview & Edit Sample", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        sample_selector = gr.Slider(0, 0, 0, step=1, label="Sample #")
                        preview_audio = gr.Audio(label="Preview", type="filepath", interactive=False)
                        preview_filename = gr.Textbox(label="Filename", interactive=False)

                    with gr.Column(scale=2):
                        with gr.Row(elem_classes="compact-row"):
                            edit_caption = gr.Textbox(label="Caption", lines=2, scale=2)
                            edit_genre = gr.Textbox(label="Genre", lines=1, scale=1)

                        with gr.Row(elem_classes="compact-row"):
                            edit_bpm = gr.Number(label="BPM", precision=0)
                            edit_keyscale = gr.Textbox(label="Key")
                            edit_timesig = gr.Dropdown(["", "2", "3", "4", "6"], label="Time Sig")
                            edit_duration = gr.Number(label="Duration", precision=1, interactive=False)

                        with gr.Row(elem_classes="compact-row"):
                            edit_lyrics = gr.Textbox(label="Lyrics", lines=2, placeholder="[Verse]\nLyrics...")
                            prompt_override = gr.Dropdown(["Use Global Ratio", "Caption", "Genre"], value="Use Global Ratio", label="Prompt Type")

                        with gr.Row(elem_classes="compact-row"):
                            edit_language = gr.Dropdown(["instrumental", "en", "zh", "ja", "ko", "es", "unknown"], value="instrumental", label="Language")
                            edit_instrumental = gr.Checkbox(label="Instrumental", value=True)
                            save_edit_btn = gr.Button("💾 Save", variant="secondary")

                        edit_status = gr.Textbox(label="Status", interactive=False)

            # Save Dataset
            with gr.Row():
                with gr.Row(elem_classes="compact-row", scale=3):
                    save_path = gr.Textbox(label="Save Path", value="./datasets/my_lora_dataset.json", scale=4)
                    save_path_picker = gr.Button("📁", scale=0, min_width=45)
                save_dataset_btn = gr.Button("💾 Save Dataset", variant="primary", scale=1)
            save_status = gr.Textbox(label="Save Status", interactive=False)

        # ==================== STEP 3: PREPROCESS ====================
        with gr.Tab("③ Preprocess", id="tab_preprocess"):
            gr.HTML('<div class="section-title">⚡ Convert to Training Tensors</div>')

            gr.HTML("""
            <div class="info-box">
                Preprocessing converts your labeled audio into optimized tensors for fast training:<br>
                • Encodes audio → VAE latents<br>
                • Encodes text → embeddings<br>
                • Saves as .pt files
            </div>
            """)

            with gr.Row(elem_classes="compact-row"):
                preprocess_output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./datasets/preprocessed_tensors",
                    scale=3,
                )
                preprocess_dir_picker = gr.Button("📁", scale=0, min_width=45)
                max_duration_slider = gr.Slider(60, 600, 240, step=30, label="Max Duration (s)", scale=2)

            preprocess_btn = gr.Button("⚡ Start Preprocessing", variant="primary", size="lg", elem_classes="primary-action")
            preprocess_progress = gr.Textbox(label="Progress", interactive=False, lines=3)

        # ==================== STEP 4: TRAIN ====================
        with gr.Tab("④ Train", id="tab_train"):
            # GPU Preset selector
            with gr.Row(elem_classes="compact-row"):
                gpu_preset = gr.Dropdown(
                    choices=list(GPU_PRESETS.keys()),
                    value="Custom",
                    label="🎮 GPU Preset",
                    info="Auto-configure all training params for your GPU. Select 'Custom' to set manually.",
                    scale=2,
                )
                gr.HTML('<div style="padding:6px 10px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:8px;display:flex;align-items:center"><span style="font-size:12px;color:#16a34a">💡 Select your GPU to auto-fill optimal settings. You can still fine-tune individual params after.</span></div>', scale=3)

            with gr.Row():
                # Left: Dataset & LoRA Settings
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📊 Training Data</div>')

                    with gr.Row(elem_classes="compact-row"):
                        training_tensor_dir = gr.Textbox(
                            label="Tensors Directory",
                            value="./datasets/preprocessed_tensors",
                            scale=4,
                        )
                        tensor_dir_picker = gr.Button("📁", scale=0, min_width=45)

                    load_tensors_btn = gr.Button("📂 Load", variant="secondary", size="sm")
                    training_dataset_info = gr.Textbox(label="Dataset Info", interactive=False, lines=2)

                    gr.HTML('<div class="section-title" style="margin-top:15px">⚙️ LoRA Settings</div>')

                    lora_rank = gr.Slider(4, 256, 64, step=4, label="Rank (r)", info="64 recommended (32 for low VRAM)")
                    lora_alpha = gr.Slider(4, 512, 128, step=4, label="Alpha", info="2x rank (α/r=2.0) recommended")
                    lora_dropout = gr.Slider(0.0, 0.5, 0.1, step=0.05, label="Dropout")

                # Right: Training Params
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">🎛️ Training Parameters</div>')

                    with gr.Row(elem_classes="compact-row"):
                        learning_rate = gr.Number(label="Learning Rate", value=1e-4, info="1e-4 recommended (ignored by Prodigy)")
                        max_epochs = gr.Number(label="Max Epochs", value=200, precision=0, info="Stop when loss plateaus (~150-200 for small datasets)")

                    with gr.Row(elem_classes="compact-row"):
                        batch_size = gr.Slider(1, 8, 2, step=1, label="Batch Size", info="2 for 24GB VRAM")
                        gradient_accumulation = gr.Slider(1, 16, 1, step=1, label="Grad Accum")

                    with gr.Row(elem_classes="compact-row"):
                        optimizer_type = gr.Dropdown(
                            choices=["adamw", "adamw8bit", "adafactor", "prodigy"],
                            value="adamw",
                            label="Optimizer",
                            info="AdamW=default, 8bit=less VRAM, Prodigy=auto LR",
                        )
                        scheduler_type = gr.Dropdown(
                            choices=["cosine", "linear", "constant", "constant_with_warmup"],
                            value="cosine",
                            label="Scheduler",
                            info="Cosine=default. Forced to constant with Prodigy",
                        )

                    with gr.Row(elem_classes="compact-row"):
                        attention_type = gr.Radio(
                            choices=["both", "self", "cross"],
                            value="both",
                            label="Attention Target",
                            info="Which attention layers to train: self, cross, or both",
                        )

                    with gr.Row(elem_classes="compact-row"):
                        model_type_radio = gr.Radio(
                            choices=["turbo", "base"],
                            value="turbo",
                            label="Model Type",
                            info="Auto-detected from loaded model. Turbo=fast (8 steps), Base=quality (continuous)",
                            interactive=False,
                        )

                    with gr.Row(elem_classes="compact-row"):
                        shift = gr.Slider(1.0, 5.0, 3.0, step=0.5, label="Shift", info="3.0=Turbo, 1.0=Base")
                        seed = gr.Number(label="Seed", value=42, precision=0)

                    # Base model parameters — hidden by default, shown when base model detected
                    base_params_group = gr.Group(visible=False)
                    with base_params_group:
                        gr.HTML('<div style="padding:6px 10px;background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);border-radius:8px;margin-bottom:8px"><span style="font-size:12px;color:#ea580c">🎯 <b>Base Model Parameters</b> — Classifier-Free Guidance for continuous timestep training</span></div>')
                        with gr.Row(elem_classes="compact-row"):
                            guidance_scale = gr.Slider(0.0, 15.0, 7.0, step=0.5, label="CFG Scale", info="Guidance strength. 5-10 recommended. 0=disabled")
                            cfg_dropout_prob = gr.Slider(0.0, 0.3, 0.1, step=0.05, label="CFG Dropout", info="Probability of dropping condition during training")
                            num_inference_steps = gr.Slider(20, 100, 60, step=10, label="Inference Steps", info="More steps = better quality, slower training")

                    save_every_n_epochs = gr.Number(label="Save Every N Epochs", value=50, precision=0, info="More checkpoints to avoid overfitting")

                    with gr.Row(elem_classes="compact-row"):
                        early_stop_enabled = gr.Checkbox(label="Early Stop", value=False, scale=1, info="Stop if loss doesn't improve")
                        early_stop_patience_val = gr.Number(label="Patience (epochs)", value=80, precision=0, scale=1, info="Epochs without improvement before stopping")

                    auto_save_best_after = gr.Number(label="Auto-Save Best After (epochs)", value=200, precision=0, info="Auto-save best model after this many warmup epochs. Early stop also starts here. 0=disabled.")

                    max_latent_length = gr.Slider(0, 6000, 1500, step=500, label="Max Crop Length (latent frames)", info="Random-crop long audio each epoch. 1500≈60s (default, matches inference). 0=full length. Shorter = much faster (attention is O(T²))")

                    with gr.Row(elem_classes="compact-row"):
                        gradient_checkpointing_enabled = gr.Checkbox(label="Gradient Checkpointing", value=False, info="Save ~40-60% VRAM at ~30% speed cost. Recommended for ≤16GB VRAM.")
                        encoder_offloading_enabled = gr.Checkbox(label="Encoder Offloading", value=False, info="Move encoder to CPU during training. Saves ~2-4 GB VRAM.")

                    torch_compile_enabled = gr.Checkbox(label="torch.compile (Experimental)", value=False, info="JIT-compile decoder. Off by default — slow first epoch. Only enable for large datasets on Linux.")

                    with gr.Row(elem_classes="compact-row"):
                        lora_output_dir = gr.Textbox(label="Output Dir", value="./lora_output", scale=4)
                        lora_output_picker = gr.Button("📁", scale=0, min_width=45)

            # Resume from checkpoint
            with gr.Accordion("🔄 Resume from Checkpoint", open=False):
                resume_enabled = gr.Checkbox(label="Resume training from a saved checkpoint", value=False)
                with gr.Row(elem_classes="compact-row"):
                    resume_dir = gr.Textbox(label="Previous Training Output", value="./lora_output", scale=4, info="Folder containing checkpoints/ from a previous run")
                    resume_dir_picker = gr.Button("📁", scale=0, min_width=45)
                with gr.Row(elem_classes="compact-row"):
                    resume_checkpoint_dropdown = gr.Dropdown(label="Checkpoint to Resume From", choices=[], scale=3, info="Select which epoch to continue from")
                    resume_scan_btn = gr.Button("🔍 Scan", variant="secondary", size="sm", scale=1)
                resume_scan_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)

            # Gradient Sensitivity Estimation
            with gr.Accordion("🔬 Gradient Sensitivity Estimation", open=False):
                gr.HTML("""
                <div style="padding:8px 12px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:8px;margin-bottom:10px">
                    <span style="font-size:12px;color:#2563eb">
                        <b>🔬 Layer Importance Analysis</b> — Runs a few forward/backward passes on your dataset
                        to measure which attention layers are most sensitive. Use the results to select targeted
                        attention layers for training (self vs cross) or to understand your dataset better.
                    </span>
                </div>
                """)
                with gr.Row(elem_classes="compact-row"):
                    estimate_tensor_dir = gr.Textbox(label="Tensors Directory", value="./datasets/preprocessed_tensors", scale=4)
                    estimate_dir_picker = gr.Button("📁", scale=0, min_width=45)
                with gr.Row(elem_classes="compact-row"):
                    estimate_batches = gr.Slider(1, 50, 10, step=1, label="Batches to Evaluate", info="More batches = more accurate but slower")
                    estimate_granularity = gr.Radio(choices=["layer", "module"], value="layer", label="Granularity", info="Layer=attention blocks, Module=individual q/k/v/o projections")
                    estimate_top_k = gr.Slider(5, 50, 20, step=5, label="Top K Results")
                with gr.Row():
                    estimate_btn = gr.Button("🔬 Run Estimation", variant="secondary", size="lg")
                estimate_progress = gr.Textbox(label="Progress", interactive=False)
                estimate_results = gr.Dataframe(
                    headers=["Rank", "Module", "Score", "Raw Norm"],
                    label="Results (sorted by importance)",
                    interactive=False,
                )

            # Training Controls
            gr.HTML('<div class="section-title" style="margin-top:20px">🚀 Training</div>')

            with gr.Row():
                start_training_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", scale=2, elem_classes="primary-action")
                stop_training_btn = gr.Button("⏹️ Stop", variant="stop", size="lg", scale=1)

            training_progress = gr.Textbox(label="Progress", interactive=False)

            with gr.Row():
                training_log = gr.Textbox(label="Log", interactive=False, lines=8, scale=1)
                training_loss_plot = gr.LinePlot(x="step", y="loss", title="Loss", x_title="Step", y_title="Loss", scale=1)

            # Export
            gr.HTML('<div class="section-title" style="margin-top:20px">📦 Export LoRA</div>')

            with gr.Row(elem_classes="compact-row"):
                export_path = gr.Textbox(label="Export Path", value="./checkpoints/my_custom_lora", scale=3)
                export_path_picker = gr.Button("📁", scale=0, min_width=45)
                export_lora_btn = gr.Button("📦 Export", variant="secondary", scale=1)

            export_status = gr.Textbox(label="Export Status", interactive=False)

        # ==================== STEP 5: MERGE LoRA ====================
        with gr.Tab("⑤ Merge", id="tab_merge"):
            gr.HTML("""
            <div class="info-box">
                <strong>🔀 Merge LoRA into Base Model</strong><br>
                Permanently integrates LoRA adapter weights into the base ACE-Step model, producing a standalone
                safetensors file. Use this when:<br>
                • ComfyUI or other tools don't support LoRA adapters natively<br>
                • You want a single model file without adapter overhead<br>
                • You want to share a ready-to-use fine-tuned model<br><br>
                ⚠️ Requires ~10 GB RAM to load the base model + merge. The service does NOT need to be loaded.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📥 Input</div>')

                    merge_base_model = gr.Dropdown(
                        label="Base Model Checkpoint",
                        choices=list_available_checkpoints(),
                        value=list_available_checkpoints()[0] if list_available_checkpoints() else None,
                        info="The original ACE-Step model that was used for training"
                    )
                    merge_refresh_base = gr.Button("🔄 Refresh", size="sm")

                    with gr.Row(elem_classes="compact-row"):
                        merge_lora_dir = gr.Textbox(label="LoRA Training Output", value="./lora_output", scale=4)
                        merge_lora_dir_picker = gr.Button("📁", scale=0, min_width=45)

                    with gr.Row(elem_classes="compact-row"):
                        merge_checkpoint_dropdown = gr.Dropdown(
                            label="LoRA Checkpoint",
                            choices=[],
                            info="Select which epoch/checkpoint to merge"
                        )
                        merge_scan_btn = gr.Button("🔍 Scan", variant="secondary", size="sm")
                    merge_scan_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📤 Output</div>')

                    with gr.Row(elem_classes="compact-row"):
                        merge_output_dir = gr.Textbox(
                            label="Merged Model Output Directory",
                            value=str(get_checkpoints_dir() / "acestep-v15-merged"),
                            scale=4,
                            info="The merged model will be saved here as safetensors"
                        )
                        merge_output_picker = gr.Button("📁", scale=0, min_width=45)

                    gr.HTML("""
                    <div class="quick-tip">
                        <strong>💡 Tip:</strong> Save inside <code>checkpoints/</code> with a name starting
                        with <code>acestep-v15-</code> and it will appear in the model dropdown for inference.
                    </div>
                    """)

            merge_btn = gr.Button("🔀 Merge LoRA into Base Model", variant="primary", size="lg", elem_classes="primary-action")
            merge_status = gr.Textbox(label="Merge Status", interactive=False, lines=4)

        # ==================== EVENT HANDLERS ====================

        # File Pickers
        split_input_picker.click(fn=open_folder_picker, outputs=[split_input_dir])
        split_output_picker.click(fn=open_folder_picker, outputs=[split_output_dir])
        audio_dir_picker.click(fn=open_folder_picker, outputs=[audio_directory])
        load_dataset_picker.click(fn=pick_json_file, outputs=[load_dataset_path])
        save_path_picker.click(fn=lambda: open_save_file_picker(".json"), outputs=[save_path])

        # Audio Splitter
        split_btn.click(
            fn=split_audio_files,
            inputs=[split_input_dir, split_output_dir, split_duration],
            outputs=[split_status],
        )
        preprocess_dir_picker.click(fn=open_folder_picker, outputs=[preprocess_output_dir])
        tensor_dir_picker.click(fn=open_folder_picker, outputs=[training_tensor_dir])
        lora_output_picker.click(fn=open_folder_picker, outputs=[lora_output_dir])
        export_path_picker.click(fn=open_folder_picker, outputs=[export_path])

        # Service
        refresh_btn.click(fn=lambda: gr.Dropdown(choices=list_available_checkpoints()), outputs=[checkpoint_dropdown])
        init_btn.click(fn=initialize_service, inputs=[checkpoint_dropdown], outputs=[service_status, status_bar, shift, model_type_radio, guidance_scale, cfg_dropout_prob, num_inference_steps, base_params_group])
        unload_btn.click(fn=unload_service, outputs=[service_status, status_bar])

        # Dataset
        scan_btn.click(
            fn=scan_audio_directory,
            inputs=[audio_directory, dataset_name, custom_tag, tag_position, all_instrumental, genre_ratio, dataset_builder_state],
            outputs=[audio_files_table, scan_status, sample_selector, dataset_builder_state, status_bar],
        )
        load_dataset_btn.click(
            fn=load_existing_dataset,
            inputs=[load_dataset_path, dataset_builder_state],
            outputs=[scan_status, audio_files_table, sample_selector, dataset_builder_state, status_bar],
        )
        auto_label_btn.click(
            fn=auto_label_samples,
            inputs=[skip_metas, only_unlabeled, dataset_builder_state],
            outputs=[audio_files_table, label_progress, dataset_builder_state, status_bar],
        )
        sample_selector.change(
            fn=get_sample_preview,
            inputs=[sample_selector, dataset_builder_state],
            outputs=[preview_audio, preview_filename, edit_caption, edit_genre, prompt_override, edit_lyrics, edit_bpm, edit_keyscale, edit_timesig, edit_duration, edit_language, edit_instrumental],
        )
        save_edit_btn.click(
            fn=save_sample_edit,
            inputs=[sample_selector, edit_caption, edit_genre, prompt_override, edit_lyrics, edit_bpm, edit_keyscale, edit_timesig, edit_language, edit_instrumental, dataset_builder_state],
            outputs=[audio_files_table, edit_status, dataset_builder_state],
        )
        save_dataset_btn.click(
            fn=save_dataset,
            inputs=[save_path, dataset_name, custom_tag, tag_position, dataset_builder_state],
            outputs=[save_status],
        )

        # Preprocessing (custom_tag + tag_position read directly from UI for failsafe)
        preprocess_btn.click(
            fn=preprocess_dataset,
            inputs=[preprocess_output_dir, max_duration_slider, custom_tag, tag_position, dataset_builder_state],
            outputs=[preprocess_progress, status_bar],
        )

        # Training
        load_tensors_btn.click(fn=load_tensor_dataset, inputs=[training_tensor_dir], outputs=[training_dataset_info, max_epochs, save_every_n_epochs])
        resume_dir_picker.click(fn=open_folder_picker, outputs=[resume_dir])
        resume_scan_btn.click(
            fn=scan_lora_checkpoints,
            inputs=[resume_dir],
            outputs=[resume_checkpoint_dropdown, resume_scan_status],
        )
        # GPU Presets
        gpu_preset.change(
            fn=apply_gpu_preset,
            inputs=[gpu_preset],
            outputs=[
                lora_rank, lora_alpha, lora_dropout,
                learning_rate, max_epochs, batch_size, gradient_accumulation,
                optimizer_type, scheduler_type, attention_type,
                gradient_checkpointing_enabled, encoder_offloading_enabled, torch_compile_enabled,
                early_stop_enabled, early_stop_patience_val, auto_save_best_after,
                save_every_n_epochs, max_latent_length,
            ],
        )

        # Gradient Estimation
        estimate_dir_picker.click(fn=open_folder_picker, outputs=[estimate_tensor_dir])
        estimate_btn.click(
            fn=run_gradient_estimation,
            inputs=[estimate_tensor_dir, estimate_batches, estimate_granularity, estimate_top_k],
            outputs=[estimate_progress, estimate_results],
        )

        start_training_btn.click(
            fn=start_training,
            inputs=[training_tensor_dir, lora_rank, lora_alpha, lora_dropout, learning_rate, max_epochs, batch_size, gradient_accumulation, save_every_n_epochs, shift, seed, lora_output_dir, early_stop_enabled, early_stop_patience_val, auto_save_best_after, max_latent_length, torch_compile_enabled, model_type_radio, guidance_scale, cfg_dropout_prob, num_inference_steps, optimizer_type, scheduler_type, attention_type, gradient_checkpointing_enabled, encoder_offloading_enabled, resume_enabled, resume_dir, resume_checkpoint_dropdown, training_state],
            outputs=[training_progress, training_log, training_loss_plot, training_state, status_bar],
        )
        stop_training_btn.click(fn=stop_training, inputs=[training_state], outputs=[training_progress, training_state])
        export_lora_btn.click(fn=export_lora, inputs=[export_path, lora_output_dir], outputs=[export_status])

        # Merge LoRA
        merge_refresh_base.click(fn=lambda: gr.Dropdown(choices=list_available_checkpoints()), outputs=[merge_base_model])
        merge_lora_dir_picker.click(fn=open_folder_picker, outputs=[merge_lora_dir])
        merge_output_picker.click(fn=open_folder_picker, outputs=[merge_output_dir])
        merge_scan_btn.click(
            fn=scan_lora_checkpoints,
            inputs=[merge_lora_dir],
            outputs=[merge_checkpoint_dropdown, merge_scan_status],
        )
        merge_btn.click(
            fn=merge_lora_to_safetensors,
            inputs=[merge_base_model, merge_lora_dir, merge_checkpoint_dropdown, merge_output_dir],
            outputs=[merge_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="ACE-Step LoRA Training UI")
    parser.add_argument("--port", type=int, default=7861, help="Port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    logger.info(f"Starting ACE-Step LoRA Training UI on {args.host}:{args.port}")

    demo = create_ui()
    demo.queue()

    # Allow access to all drives on Windows for audio file preview
    # This is necessary because training data can be anywhere on the system
    allowed_paths = [
        "/",  # Linux/Mac root
        "C:\\", "D:\\", "E:\\", "F:\\", "G:\\", "H:\\",  # Common Windows drives
    ]

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
