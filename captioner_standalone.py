#!/usr/bin/env python3
"""
ACE-Step Captioner - Standalone Script
Genera caption dettagliate per file audio usando il modello ACE-Step Captioner (11B).

Uso:
    python captioner_standalone.py --input_dir ./audio --output_dir ./captions
    python captioner_standalone.py --input_dir ./audio --output_csv ./dataset.csv

Il modello genera descrizioni musicali professionali includendo:
- Stile e genere musicale
- Strumenti riconosciuti
- Struttura del brano (intro, verse, chorus, etc.)
- Caratteristiche timbriche e dinamiche
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Optional, List, Dict, Callable
import csv

import torch


class AceStepCaptioner:
    """Handler per il modello ACE-Step Captioner basato su Qwen2.5 Omni."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Inizializza il captioner.

        Args:
            model_path: Path al modello ACE-Step Captioner
            device: Device da usare ("auto", "cuda", "cpu")
        """
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.device = device
        self._resolved_device = None
        # ACE-Step Transcriber for lyrics (replaces Whisper)
        self.transcriber_model = None
        self.transcriber_processor = None
        self.transcriber_path = None

    def load_model(self, progress_callback: Optional[Callable] = None) -> str:
        """
        Carica il modello Qwen2.5 Omni Captioner.

        Args:
            progress_callback: Callback opzionale per progress (step, message)

        Returns:
            Status message
        """
        try:
            if progress_callback:
                progress_callback(1, "Importing transformers...")

            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

            # Resolve device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._resolved_device = "cuda"
                else:
                    self._resolved_device = "cpu"
            else:
                self._resolved_device = self.device

            if progress_callback:
                progress_callback(2, f"Loading processor from {self.model_path}...")

            # Load processor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

            if progress_callback:
                progress_callback(3, "Loading model (this may take a while)...")

            # Determine dtype
            dtype = torch.bfloat16 if self._resolved_device == "cuda" else torch.float32

            # Load model - disable audio output (talker) to save ~2GB VRAM
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map="auto",
                enable_audio_output=False,
            )
            self.model.eval()

            if progress_callback:
                progress_callback(4, "Model loaded successfully!")

            return f"✅ Captioner loaded on {self._resolved_device}"

        except Exception as e:
            return f"❌ Failed to load model: {str(e)}"

    def caption_audio(self, audio_path: str, max_new_tokens: int = 512) -> str:
        """
        Genera caption per un singolo file audio.

        Args:
            audio_path: Path al file audio
            max_new_tokens: Massimo numero di token da generare

        Returns:
            Caption generata
        """
        if self.model is None or self.processor is None:
            return "❌ Model not loaded. Call load_model() first."

        try:
            import librosa
        except ImportError:
            return "❌ librosa not installed. Run: pip install librosa"

        try:
            # Load audio with librosa (works on all platforms, unlike torchcodec)
            # librosa.load with mono=True returns a 1D array of shape (T,)
            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

            caption = self._run_inference(
                audio_array,
                "*Task* Describe this audio in detail",
                max_new_tokens,
            )

            return caption

        except Exception as e:
            return f"❌ Error processing {audio_path}: {str(e)}"

    @staticmethod
    def _clean_model_output(text: str) -> str:
        """Remove common model output artifacts (role prefixes, prompt echoes)."""
        text = text.strip()
        # Remove role prefix "assistant\n" or "assistant " at start
        if text.lower().startswith("assistant"):
            text = text[len("assistant"):].strip()
        return text

    @staticmethod
    def _parse_key(raw: str) -> str:
        """Extract musical key from model output using cascading regex patterns."""
        import re

        # 1. Structured "key: D Minor" or "key: C# Major"
        m = re.search(r'key[:\s]+([A-Ga-g][#b♯♭]?\s*(?:natural\s+)?(?:major|minor|maj|min))', raw, re.IGNORECASE)
        if m:
            val = m.group(1).strip().title()
            return re.sub(r'\s*Natural\s+', ' ', val).strip()

        # 2. Shorthand "key: Dm" or "key: D#m"
        m = re.search(r'key[:\s]+([A-Ga-g][#b♯♭]?)\s*m\b', raw, re.IGNORECASE)
        if m:
            return m.group(1).strip().upper() + " Minor"

        # 3. "key: D" (note only) — check nearby text for major/minor
        m = re.search(r'key[:\s]+([A-Ga-g][#b♯♭]?)\b', raw, re.IGNORECASE)
        if m:
            note = m.group(1).strip().upper()
            after = raw[m.end():m.end() + 30].lower()
            if 'min' in after:
                return f"{note} Minor"
            elif 'maj' in after:
                return f"{note} Major"
            return note

        # 4. Anywhere: "D Minor", "C# Major", "Bb minor" etc.
        m = re.search(r'\b([A-G][#b♯♭]?\s+(?:major|minor|maj|min))\b', raw, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()

        # 5. Anywhere: "in D minor", "the key of A"
        m = re.search(r'(?:in|of)\s+([A-G][#b♯♭]?)\s*(major|minor|maj|min)', raw, re.IGNORECASE)
        if m:
            note = m.group(1).strip().upper()
            mode = "Major" if "maj" in m.group(2).lower() else "Minor"
            return f"{note} {mode}"

        return ""

    @staticmethod
    def _parse_genre(raw_lower: str) -> str:
        """Extract genre from model output (lowercased)."""
        import re

        # 1. Structured "genre: electronic pop"
        m = re.search(r'genre[:\s]+([a-z][a-z\s\-/&]+)', raw_lower)
        if m:
            genre_val = m.group(1).strip().split('\n')[0].strip()
            if len(genre_val) > 30:
                genre_val = genre_val.split(',')[0].strip()
            # Remove trailing filler words like "music", "style"
            genre_val = re.sub(r'\s*(music|style|song|track)\s*$', '', genre_val).strip()
            if genre_val:
                return genre_val

        # 2. Fallback: check common genre keywords
        genre_keywords = [
            "electronic pop", "synth-pop", "synthpop", "hip-hop", "trip-hop",
            "lo-fi", "lo fi", "r&b", "drum and bass", "drum & bass",
            "pop", "rock", "jazz", "classical", "electronic", "edm",
            "hip hop", "rap", "rnb", "folk", "country", "blues",
            "metal", "punk", "indie", "soul", "reggae", "reggaeton",
            "latin", "ballad", "acoustic", "ambient", "house", "techno",
            "disco", "funk", "gospel", "alternative", "grunge",
        ]
        for gk in genre_keywords:
            if gk in raw_lower:
                return gk

        return ""

    def _run_inference(self, audio_array, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Run inference with a given prompt on pre-loaded audio.

        Args:
            audio_array: 1D numpy array of audio samples at 16kHz
            prompt: The text prompt for the model
            max_new_tokens: Maximum tokens to generate

        Returns:
            Cleaned model output text
        """
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "dummy"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=text,
            audio=[audio_array],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)

        # Save input length to strip prompt tokens from output
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                thinker_max_new_tokens=max_new_tokens,
                return_audio=False,
                thinker_do_sample=True,
                thinker_temperature=0.7,
                thinker_top_p=0.9,
            )

        # Strip input tokens — generate() returns full sequence (input + new tokens)
        new_tokens = output_ids[:, input_len:]

        result = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._clean_model_output(result)

    def analyze_audio_metadata(self, audio_path: str, audio_array=None, sr: int = 16000) -> dict:
        """
        Extract BPM, duration, key, time signature, and genre from audio.

        - BPM + Duration: computed with librosa (fast, no model needed)
        - Key + Time Signature + Genre: extracted by Qwen from audio

        Args:
            audio_path: Path to audio file
            audio_array: Pre-loaded audio array (optional, avoids reloading)
            sr: Sample rate of audio_array

        Returns:
            Dict with keys: bpm, duration, keyscale, timesignature, genre
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa not installed. Run: pip install librosa")
        import numpy as np

        metadata = {
            "bpm": None,
            "duration": 0,
            "keyscale": "",
            "timesignature": "",
            "genre": "",
        }

        try:
            # Load audio if not provided
            if audio_array is None:
                audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

            # --- Duration (from sample count) ---
            metadata["duration"] = int(len(audio_array) / sr)

            # --- BPM (librosa beat tracking) ---
            # Use native sr for better beat detection, but 16kHz works too
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0])
            metadata["bpm"] = int(round(tempo))

            print(f"  [Metadata] Duration: {metadata['duration']}s, BPM: {metadata['bpm']}")

        except Exception as e:
            print(f"  [Metadata] librosa analysis error: {e}")

        # --- Key, Time Signature, Genre via Qwen ---
        if self.model is not None and self.processor is not None:
            try:
                import re
                prompt = (
                    "Analyze this audio and respond with EXACTLY these three lines:\n"
                    "key: <note> <Major or Minor>\n"
                    "time_signature: <number>\n"
                    "genre: <genre>\n\n"
                    "Example:\n"
                    "key: D Minor\n"
                    "time_signature: 4\n"
                    "genre: electronic pop"
                )
                raw = self._run_inference(audio_array, prompt, max_new_tokens=100)
                print(f"  [Metadata] Qwen raw: {repr(raw[:300])}")

                raw_lower = raw.lower()

                # --- Parse key ---
                metadata["keyscale"] = self._parse_key(raw)

                # --- Parse time signature ---
                ts_match = re.search(r'time.?signature[:\s]+(\d)', raw_lower)
                if ts_match:
                    metadata["timesignature"] = ts_match.group(1)
                else:
                    # Fallback: look for "X/Y" pattern
                    ts_match2 = re.search(r'(\d)/\d\s*(?:time|meter|beat)', raw_lower)
                    if ts_match2:
                        metadata["timesignature"] = ts_match2.group(1)
                    else:
                        # Default to 4 for most pop/rock music
                        metadata["timesignature"] = "4"

                # --- Parse genre ---
                metadata["genre"] = self._parse_genre(raw_lower)

                print(f"  [Metadata] Key: {metadata['keyscale']}, TimeSig: {metadata['timesignature']}, Genre: {metadata['genre']}")

            except Exception as e:
                print(f"  [Metadata] Qwen analysis error: {e}")

        return metadata

    def transcribe_lyrics(self, audio_path: str, max_new_tokens: int = 1024, language: str = "auto") -> tuple:
        """
        Transcribe lyrics using ACE-Step Transcriber.
        Produces structured lyrics with section tags ([Verse], [Chorus], etc.)
        and auto-detects the language.

        Args:
            audio_path: Path to the audio file
            max_new_tokens: Maximum tokens to generate
            language: Not used (transcriber auto-detects), kept for API compatibility

        Returns:
            Tuple of (lyrics_text, detected_language)
            lyrics_text: Structured lyrics with section tags, or "[Instrumental]"
            detected_language: Detected language code (e.g. "it", "en")
        """
        try:
            if self.transcriber_model is None:
                return "❌ Transcriber not loaded. Load model first.", "unknown"

            try:
                import librosa
            except ImportError:
                return "❌ librosa not installed. Run: pip install librosa", "unknown"

            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

            print(f"  [Transcriber] Processing {Path(audio_path).name} ({len(audio_array)/sr:.1f}s)...")
            raw_output = self._transcriber_inference(audio_array, max_new_tokens=max_new_tokens)
            print(f"  [Transcriber] Raw output ({len(raw_output)} chars):\n    {raw_output[:500]}")

            lyrics, detected_lang = self._parse_transcriber_output(raw_output)
            print(f"  [Transcriber] Language: {detected_lang}, Lyrics length: {len(lyrics)} chars")

            return lyrics, detected_lang

        except Exception as e:
            return f"❌ Error transcribing {audio_path}: {str(e)}", "unknown"

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        output_csv: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        max_new_tokens: int = 512,
    ) -> List[Dict]:
        """
        Processa tutti i file audio in una cartella.

        Args:
            input_dir: Cartella con file audio
            output_dir: Cartella per salvare JSON individuali (opzionale)
            output_csv: File CSV per salvare tutte le caption (opzionale)
            progress_callback: Callback (current, total, filename)
            max_new_tokens: Max tokens per caption

        Returns:
            Lista di risultati [{filename, path, caption}, ...]
        """
        # Find audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
        input_path = Path(input_dir)

        audio_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in audio_extensions
        ])

        if not audio_files:
            print(f"⚠️ No audio files found in {input_dir}")
            return []

        results = []

        for i, audio_path in enumerate(audio_files):
            if progress_callback:
                progress_callback(i + 1, len(audio_files), audio_path.name)

            # Generate caption
            caption = self.caption_audio(str(audio_path), max_new_tokens)

            result = {
                "filename": audio_path.name,
                "path": str(audio_path.absolute()),
                "caption": caption
            }
            results.append(result)

            # Save individual JSON
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                json_path = Path(output_dir) / f"{audio_path.stem}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            # Print caption preview
            preview = caption[:100] + "..." if len(caption) > 100 else caption
            print(f"   → {preview}")

        # Save CSV
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "path", "caption"])
                writer.writeheader()
                writer.writerows(results)

        return results

    def load_transcriber(self, transcriber_path: str = "ACE-Step/acestep-transcriber") -> str:
        """
        Load ACE-Step Transcriber model for structured lyrics transcription.
        Uses Qwen2.5-Omni architecture, fine-tuned for music transcription.
        Outputs structured lyrics with section tags ([Verse], [Chorus], etc.)

        Args:
            transcriber_path: Path or HuggingFace model ID for ACE-Step Transcriber

        Returns:
            Status message
        """
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

            device = self._resolved_device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            print(f"  [Transcriber] Loading from {transcriber_path}...")
            self.transcriber_processor = Qwen2_5OmniProcessor.from_pretrained(transcriber_path)
            self.transcriber_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                transcriber_path,
                dtype=dtype,
                device_map="auto",
                enable_audio_output=False,
            )
            self.transcriber_model.eval()
            self.transcriber_path = transcriber_path

            return f"✅ Transcriber loaded on {device}"
        except Exception as e:
            return f"❌ Failed to load Transcriber: {str(e)}"

    def _transcriber_inference(self, audio_array, max_new_tokens: int = 1024) -> str:
        """
        Run ACE-Step Transcriber on pre-loaded audio.

        Args:
            audio_array: 1D numpy array of audio samples at 16kHz
            max_new_tokens: Maximum tokens to generate

        Returns:
            Raw transcriber output text
        """
        if self.transcriber_model is None or self.transcriber_processor is None:
            return ""

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "dummy"},
                    {"type": "text", "text": "*Task* Transcribe this audio in detail"},
                ],
            }
        ]

        text = self.transcriber_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.transcriber_processor(
            text=text,
            audio=[audio_array],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.transcriber_model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.transcriber_model.generate(
                **inputs,
                thinker_max_new_tokens=max_new_tokens,
                return_audio=False,
                thinker_do_sample=False,
            )

        new_tokens = output_ids[:, input_len:]

        result = self.transcriber_processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._clean_model_output(result)

    @staticmethod
    def _parse_transcriber_output(raw_output: str) -> tuple:
        """
        Parse the structured output from ACE-Step Transcriber.

        Expected format:
            # Languages
            en

            # Lyrics
            [Verse 1]
            Walking down the empty street tonight
            ...

        Returns:
            Tuple of (lyrics_text, detected_language_code)
        """
        import re

        language = "unknown"
        lyrics = ""

        if not raw_output or not raw_output.strip():
            return "[Instrumental]", "unknown"

        text = raw_output.strip()

        # Extract language from "# Languages" section
        lang_match = re.search(r'#\s*Languages?\s*\n\s*(\w+)', text, re.IGNORECASE)
        if lang_match:
            language = lang_match.group(1).strip().lower()

        # Extract lyrics from "# Lyrics" section
        lyrics_match = re.search(r'#\s*Lyrics?\s*\n(.*)', text, re.DOTALL | re.IGNORECASE)
        if lyrics_match:
            lyrics = lyrics_match.group(1).strip()
        else:
            # Fallback: if no "# Lyrics" header, check if there are section tags directly
            if re.search(r'\[(Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Post-Chorus|Instrumental|Spoken)', text, re.IGNORECASE):
                # Find from first section tag onwards
                section_match = re.search(r'(\[(?:Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Post-Chorus|Instrumental|Spoken|Guitar|Piano|Interlude).*)', text, re.DOTALL | re.IGNORECASE)
                if section_match:
                    lyrics = section_match.group(1).strip()
            else:
                # No structure found — treat entire output as lyrics
                # Remove the language header if present
                cleaned = re.sub(r'#\s*Languages?\s*\n\s*\w+\s*\n?', '', text).strip()
                if cleaned:
                    lyrics = f"[Verse]\n{cleaned}"

        # Check if the lyrics indicate instrumental
        if not lyrics or lyrics.strip().lower() in ("[instrumental]", "instrumental", ""):
            return "[Instrumental]", language

        # Clean up excessive blank lines
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)

        return lyrics, language

    def unload_model(self):
        """Libera VRAM e memoria."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.unload_transcriber()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ All models unloaded, memory freed")

    def unload_transcriber(self):
        """Unload only the transcriber model to free VRAM."""
        if self.transcriber_model is not None:
            del self.transcriber_model
            self.transcriber_model = None

        if self.transcriber_processor is not None:
            del self.transcriber_processor
            self.transcriber_processor = None

        self.transcriber_path = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


DEFAULT_MODEL_PATH = "ACE-Step/acestep-captioner"
DEFAULT_TRANSCRIBER_PATH = "ACE-Step/acestep-transcriber"


# ============== Gradio UI ==============

def open_folder_picker() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title="Select Folder")
        root.destroy()
        return folder_path if folder_path else ""
    except Exception:
        return ""


def split_audio_files_ui(input_dir, output_dir, segment_duration, progress=None):
    """Split long audio files into shorter segments."""
    if not input_dir or not input_dir.strip():
        return "❌ Please select an input directory"
    if not output_dir or not output_dir.strip():
        return "❌ Please select an output directory"

    input_dir = input_dir.strip()
    output_dir = output_dir.strip()

    if not os.path.exists(input_dir):
        return f"❌ Input directory not found: {input_dir}"

    os.makedirs(output_dir, exist_ok=True)

    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    audio_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in audio_extensions
    ]

    if not audio_files:
        return f"❌ No audio files found in {input_dir}"

    try:
        from pydub import AudioSegment
    except ImportError:
        return "❌ pydub not installed. Run: pip install pydub"

    total_segments = 0
    processed_files = 0
    segment_ms = int(segment_duration) * 1000

    for i, audio_path in enumerate(audio_files):
        if progress:
            progress((i + 1) / len(audio_files), desc=f"Splitting {Path(audio_path).name}...")
        try:
            audio = AudioSegment.from_file(audio_path)
            base_name = Path(audio_path).stem
            segment_idx = 0
            start_ms = 0

            while start_ms < len(audio):
                end_ms = min(start_ms + segment_ms, len(audio))
                if (end_ms - start_ms) >= 10000:
                    segment = audio[start_ms:end_ms]
                    segment_filename = f"{base_name}_seg{segment_idx:02d}.wav"
                    segment.export(os.path.join(output_dir, segment_filename), format="wav")
                    total_segments += 1
                    segment_idx += 1
                start_ms = end_ms

            processed_files += 1
        except Exception as e:
            continue

    return f"✅ Split {processed_files} files into {total_segments} segments ({segment_duration}s each) → {output_dir}"


# Global captioner instance for UI
_ui_captioner = None
_ui_results = []


def load_model_ui(model_path, progress=None):
    """Load the captioner model (Qwen2.5-Omni for captions + metadata)."""
    global _ui_captioner

    if not model_path or not model_path.strip():
        return "❌ Please specify model path"

    model_path = model_path.strip()
    # Accept both local paths and HuggingFace model IDs (e.g. "ACE-Step/acestep-captioner")
    # HF IDs look like "org/model" — no backslashes, no absolute path indicators
    is_hf_id = "/" in model_path and "\\" not in model_path and not os.path.isabs(model_path)
    if not is_hf_id and not os.path.exists(model_path):
        return f"❌ Model path not found: {model_path}\nTo auto-download from HuggingFace, use format: ACE-Step/acestep-captioner"

    if is_hf_id and progress:
        progress(0.1, desc=f"Downloading {model_path} from HuggingFace (this may take a while on first use)...")

    def progress_cb(step, message):
        if progress:
            progress(step / 5, desc=message)

    _ui_captioner = AceStepCaptioner(model_path)
    status = _ui_captioner.load_model(progress_cb)

    return status


def load_transcriber_ui(transcriber_path, model_path=None, progress=None):
    """Load the ACE-Step Transcriber model for structured lyrics transcription."""
    global _ui_captioner

    if _ui_captioner is None:
        # Create captioner instance without loading the captioner model
        # This enables the "transcribe only" workflow
        _ui_captioner = AceStepCaptioner(model_path or DEFAULT_MODEL_PATH)
        # Resolve device for transcriber
        if torch.cuda.is_available():
            _ui_captioner._resolved_device = "cuda"
        else:
            _ui_captioner._resolved_device = "cpu"

    if not transcriber_path or not transcriber_path.strip():
        return "❌ Please specify transcriber model path or HuggingFace ID"

    transcriber_path = transcriber_path.strip()

    if progress:
        progress(0.2, desc="Loading ACE-Step Transcriber (this may download ~22GB on first use)...")

    status = _ui_captioner.load_transcriber(transcriber_path)

    return status


def unload_transcriber_ui():
    """Unload only the transcriber to free VRAM."""
    global _ui_captioner
    if _ui_captioner is not None and _ui_captioner.transcriber_model is not None:
        _ui_captioner.unload_transcriber()
        return "✅ Transcriber unloaded, VRAM freed"
    return "⚠️ No transcriber loaded"


def unload_model_ui():
    """Unload the captioner model."""
    global _ui_captioner
    if _ui_captioner is not None:
        _ui_captioner.unload_model()
        _ui_captioner = None
        return "✅ Model unloaded, memory freed"
    return "⚠️ No model loaded"


def start_captioning_ui(input_dir, activation_tag, generate_lyrics, save_csv, csv_path, max_tokens, progress=None):
    """Run captioning, metadata extraction, and optionally lyrics transcription.

    VRAM-aware workflow:
    - If both models are loaded → runs captions + lyrics per-file
    - If only captioner loaded + lyrics requested → runs all captions first,
      then prompts to load transcriber (user must load manually and re-run,
      or we do a two-pass approach automatically if transcriber_path is set)
    """
    global _ui_captioner, _ui_results

    # Failsafe: activation tag is mandatory
    tag = activation_tag.strip() if activation_tag else ""
    if not tag:
        return "❌ Activation Tag is empty! Set a unique trigger word (e.g., LNKPRK_style) before captioning.", [], 0

    if _ui_captioner is None or _ui_captioner.model is None:
        return "❌ Load Captioner model first", [], 0

    if generate_lyrics and _ui_captioner.transcriber_model is None:
        return "❌ Load Transcriber model first (needed for lyrics generation). Tip: you can run without lyrics first, then load transcriber and re-run with lyrics enabled.", [], 0

    if not input_dir or not input_dir.strip():
        return "❌ Please select an input directory", [], 0

    input_dir = input_dir.strip()
    if not os.path.exists(input_dir):
        return f"❌ Directory not found: {input_dir}", [], 0

    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    input_path = Path(input_dir)
    audio_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in audio_extensions
    ])

    if not audio_files:
        return f"❌ No audio files found in {input_dir}", [], 0

    _ui_results = []
    max_tokens = int(max_tokens) if max_tokens else 512
    # Steps per file: caption + metadata + (lyrics if enabled)
    steps_per_file = 3 if generate_lyrics else 2
    total_steps = len(audio_files) * steps_per_file
    current_step = 0

    try:
        import librosa
    except ImportError:
        return "❌ librosa not installed. Run: pip install librosa", [], 0

    for i, audio_path in enumerate(audio_files):
        # Check if JSON already exists — load existing data to preserve fields
        json_path = input_path / f"{audio_path.stem}.json"
        existing_data = {}
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                pass

        # Load audio ONCE per file and reuse for all steps (caption + metadata + lyrics)
        try:
            audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        except Exception as e:
            print(f"  ❌ Failed to load {audio_path.name}: {e}")
            continue

        # Step 1: Caption
        current_step += 1
        if progress:
            step_label = f"[{i+1}/{len(audio_files)}] Caption: {audio_path.name}"
            progress(current_step / total_steps, desc=step_label)

        try:
            caption = _ui_captioner._run_inference(
                audio_array,
                "*Task* Describe this audio in detail",
                max_tokens,
            )
        except Exception as e:
            caption = f"❌ Error: {e}"

        result = {
            "filename": audio_path.name,
            "path": str(audio_path.absolute()),
            "caption": caption,
        }

        # Step 2: Metadata — reuse audio_array (no re-load)
        current_step += 1
        if progress:
            step_label = f"[{i+1}/{len(audio_files)}] Metadata: {audio_path.name}"
            progress(current_step / total_steps, desc=step_label)

        metadata = _ui_captioner.analyze_audio_metadata(
            str(audio_path), audio_array=audio_array, sr=sr,
        )
        result["bpm"] = metadata["bpm"]
        result["duration"] = metadata["duration"]
        result["keyscale"] = metadata["keyscale"]
        result["timesignature"] = metadata["timesignature"]
        result["genre"] = metadata["genre"]

        # Fallback: mine the caption text for key/genre if metadata extraction missed them
        if not result["keyscale"] and caption and not caption.startswith("❌"):
            result["keyscale"] = AceStepCaptioner._parse_key(caption)
            if result["keyscale"]:
                print(f"  [Metadata] Key recovered from caption: {result['keyscale']}")
        if not result["genre"] and caption and not caption.startswith("❌"):
            result["genre"] = AceStepCaptioner._parse_genre(caption.lower())
            if result["genre"]:
                print(f"  [Metadata] Genre recovered from caption: {result['genre']}")

        # Step 3: Lyrics + Language via ACE-Step Transcriber (if enabled)
        if generate_lyrics:
            current_step += 1
            if progress:
                step_label = f"[{i+1}/{len(audio_files)}] Transcribing lyrics: {audio_path.name}"
                progress(current_step / total_steps, desc=step_label)

            lyrics, detected_lang = _ui_captioner.transcribe_lyrics(str(audio_path), max_new_tokens=1024)
            result["lyrics"] = lyrics
            result["language"] = detected_lang
            result["is_instrumental"] = (lyrics.strip() == "[Instrumental]")
        else:
            # Preserve existing lyrics/language if re-running without lyrics
            result["lyrics"] = existing_data.get("lyrics", "")
            result["language"] = existing_data.get("language", "unknown")
            result["is_instrumental"] = existing_data.get("is_instrumental", True)

        # Always include activation tag
        result["custom_tag"] = tag

        _ui_results.append(result)

        # Save JSON in same folder as audio (compatible with training UI)
        json_path = input_path / f"{audio_path.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Free GPU memory between files to prevent OOM on large batches
        del audio_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV if requested
    if save_csv and csv_path and csv_path.strip():
        csv_dir = os.path.dirname(csv_path.strip())
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        fieldnames = ["filename", "path", "caption", "bpm", "keyscale", "timesignature", "duration", "genre", "language"]
        if generate_lyrics:
            fieldnames.append("lyrics")
        with open(csv_path.strip(), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(_ui_results)

    # Build table data
    table_data = []
    for i, r in enumerate(_ui_results):
        preview = r["caption"][:80] + "..." if len(r["caption"]) > 80 else r["caption"]
        bpm_str = str(r.get("bpm", "")) if r.get("bpm") else ""
        key_str = r.get("keyscale", "")
        meta_str = f"BPM:{bpm_str} Key:{key_str} {r.get('genre', '')}"
        table_data.append([i + 1, r["filename"], preview, meta_str.strip()])

    slider_max = max(0, len(_ui_results) - 1)

    lyrics_info = " + lyrics" if generate_lyrics else ""
    status = f"✅ Captioned{lyrics_info} + metadata for {len(_ui_results)} files. JSON saved in {input_dir}"
    if save_csv and csv_path:
        status += f"\n📄 CSV saved to {csv_path.strip()}"

    import gradio as gr
    return status, table_data, gr.Slider(maximum=slider_max, value=0)


def transcribe_only_ui(input_dir, activation_tag, progress=None):
    """Run ONLY lyrics transcription on files that already have caption JSONs.
    This allows a two-pass workflow to save VRAM:
    1. Load Captioner → run captions + metadata → Unload Captioner
    2. Load Transcriber → run this function → Unload Transcriber
    """
    global _ui_captioner, _ui_results

    # Failsafe: activation tag is mandatory
    tag = activation_tag.strip() if activation_tag else ""
    if not tag:
        return "❌ Activation Tag is empty! Set a tag before transcribing.", [], 0

    if _ui_captioner is None or _ui_captioner.transcriber_model is None:
        return "❌ Load Transcriber model first", [], 0

    if not input_dir or not input_dir.strip():
        return "❌ Please select an input directory", [], 0

    input_dir = input_dir.strip()
    if not os.path.exists(input_dir):
        return f"❌ Directory not found: {input_dir}", [], 0

    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    input_path = Path(input_dir)
    audio_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in audio_extensions
    ])

    if not audio_files:
        return f"❌ No audio files found in {input_dir}", [], 0

    _ui_results = []
    total = len(audio_files)

    for i, audio_path in enumerate(audio_files):
        if progress:
            progress((i + 1) / total, desc=f"[{i+1}/{total}] Transcribing: {audio_path.name}")

        # Load existing JSON if present
        json_path = input_path / f"{audio_path.stem}.json"
        result = {}
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            except Exception:
                pass

        # Ensure basic fields
        result.setdefault("filename", audio_path.name)
        result.setdefault("path", str(audio_path.absolute()))
        result.setdefault("caption", "")

        # Transcribe lyrics
        lyrics, detected_lang = _ui_captioner.transcribe_lyrics(str(audio_path), max_new_tokens=1024)
        result["lyrics"] = lyrics
        result["language"] = detected_lang
        result["is_instrumental"] = (lyrics.strip() == "[Instrumental]")
        result["custom_tag"] = tag

        _ui_results.append(result)

        # Update JSON on disk
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Build table data
    table_data = []
    for i, r in enumerate(_ui_results):
        lyrics_preview = r.get("lyrics", "")[:60] + "..." if len(r.get("lyrics", "")) > 60 else r.get("lyrics", "")
        table_data.append([i + 1, r["filename"], r.get("caption", "")[:50] + "...", lyrics_preview])

    import gradio as gr
    slider_max = max(0, len(_ui_results) - 1)
    return f"✅ Transcribed lyrics for {len(_ui_results)} files. JSON updated in {input_dir}", table_data, gr.Slider(maximum=slider_max, value=0)


def get_caption_preview(sample_idx):
    """Get preview for a captioned sample."""
    global _ui_results

    if not _ui_results:
        return None, "", "", "", None, "", "", "", "", ""

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_ui_results):
        return None, "", "", "", None, "", "", "", "", ""

    r = _ui_results[idx]
    lyrics = r.get("lyrics", "")
    bpm = r.get("bpm", None)
    keyscale = r.get("keyscale", "")
    timesig = r.get("timesignature", "")
    duration = r.get("duration", 0)
    genre = r.get("genre", "")
    language = r.get("language", "unknown")
    return r["path"], r["filename"], r["caption"], lyrics, bpm, keyscale, timesig, str(duration), genre, language


def save_caption_edit(sample_idx, new_caption, new_lyrics, new_bpm, new_key, new_timesig, new_genre, new_language):
    """Save edited caption, lyrics, and metadata back to JSON file."""
    global _ui_results

    if not _ui_results:
        return "❌ No results to edit"

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_ui_results):
        return "❌ Invalid sample index"

    _ui_results[idx]["caption"] = new_caption
    if new_lyrics is not None:
        _ui_results[idx]["lyrics"] = new_lyrics
        _ui_results[idx]["is_instrumental"] = (new_lyrics.strip() == "[Instrumental]")
    if new_bpm is not None:
        _ui_results[idx]["bpm"] = int(new_bpm) if new_bpm else None
    if new_key is not None:
        _ui_results[idx]["keyscale"] = new_key.strip()
    if new_timesig is not None:
        _ui_results[idx]["timesignature"] = new_timesig.strip()
    if new_genre is not None:
        _ui_results[idx]["genre"] = new_genre.strip()
    if new_language is not None:
        _ui_results[idx]["language"] = new_language.strip()

    # Update JSON file on disk
    audio_path = Path(_ui_results[idx]["path"])
    json_path = audio_path.parent / f"{audio_path.stem}.json"

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_ui_results[idx], f, indent=2, ensure_ascii=False)
        return f"✅ Saved for {_ui_results[idx]['filename']}"
    except Exception as e:
        return f"❌ Failed to save: {str(e)}"


CAPTIONER_CSS = """
.status-bar {
    display: flex; justify-content: space-between; padding: 12px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px; margin-bottom: 20px; border: 1px solid #0f3460;
}
.section-title {
    font-size: 18px; font-weight: 600; color: #fff; margin-bottom: 15px;
    display: flex; align-items: center; gap: 10px;
}
.compact-row { gap: 10px !important; }
.primary-action {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important; font-weight: 600 !important; padding: 12px 24px !important;
}
.primary-action:hover { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important; }
.info-box {
    background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #a0a0c0;
}
.quick-tip {
    background: rgba(0, 217, 165, 0.1); border-left: 3px solid #00d9a5;
    padding: 10px 15px; border-radius: 0 8px 8px 0; font-size: 13px; color: #a0c0b0;
}
.app-header {
    text-align: center; padding: 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; margin-bottom: 20px; border: 1px solid #0f3460;
}
.app-header h1 {
    font-size: 28px; margin: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-header p { color: #888; margin: 8px 0 0 0; font-size: 14px; }
"""


def create_captioner_ui():
    """Create the Gradio interface for ACE-Step Captioner."""
    import gradio as gr

    with gr.Blocks(title="ACE-Step Captioner", css=CAPTIONER_CSS) as demo:

        gr.HTML("""
        <div class="app-header">
            <h1>🎙️ ACE-Step Audio Captioner</h1>
            <p>Generate detailed music descriptions • JSON output compatible with LoRA Training UI</p>
        </div>
        """)

        # ==================== CAPTIONER MODEL ====================
        gr.HTML('<div class="section-title">🔧 Captioner Model (captions + metadata)</div>')

        with gr.Row(elem_classes="compact-row"):
            model_path_input = gr.Textbox(
                label="Captioner Model Path",
                value=DEFAULT_MODEL_PATH,
                scale=4,
            )
            model_path_picker = gr.Button("📁", scale=0, min_width=45)

        with gr.Row():
            load_btn = gr.Button("🚀 Load Captioner", variant="primary", scale=2, elem_classes="primary-action")
            unload_btn = gr.Button("🗑️ Unload All", variant="stop", scale=1)

        model_status = gr.Textbox(label="Captioner Status", interactive=False, lines=1)

        # ==================== TRANSCRIBER MODEL ====================
        gr.HTML('<div class="section-title">🎤 Transcriber Model (structured lyrics)</div>')

        gr.HTML("""
        <div class="info-box">
            <strong>ACE-Step Transcriber</strong> generates structured lyrics with section tags
            ([Verse], [Chorus], [Bridge], etc.) and auto-detects language.
            First load downloads ~22GB from HuggingFace. You can also use a local path.
        </div>
        """)

        with gr.Row(elem_classes="compact-row"):
            transcriber_path_input = gr.Textbox(
                label="Transcriber Model Path or HuggingFace ID",
                value=DEFAULT_TRANSCRIBER_PATH,
                scale=4,
                info="HuggingFace ID (auto-downloads) or local path",
            )
            transcriber_path_picker = gr.Button("📁", scale=0, min_width=45)

        with gr.Row():
            load_transcriber_btn = gr.Button("🎤 Load Transcriber", variant="primary", scale=2)
            unload_transcriber_btn = gr.Button("🗑️ Unload Transcriber", variant="stop", scale=1)

        transcriber_status = gr.Textbox(label="Transcriber Status", interactive=False, lines=1)

        # ==================== SPLIT AUDIO ====================
        with gr.Accordion("✂️ Split Long Audio Files (optional)", open=False):
            gr.HTML("""
            <div class="info-box">
                <strong>⚡ Tip:</strong> Training is much faster with shorter audio segments (30-60s).
                Split your files here before captioning.
            </div>
            """)
            with gr.Row(elem_classes="compact-row"):
                split_input_dir = gr.Textbox(label="Input Folder (original audio)", placeholder="Folder with long audio files...", scale=3)
                split_input_picker = gr.Button("📁", scale=0, min_width=45)

            with gr.Row(elem_classes="compact-row"):
                split_output_dir = gr.Textbox(label="Output Folder (split segments)", placeholder="Folder to save segments...", scale=3)
                split_output_picker = gr.Button("📁", scale=0, min_width=45)

            with gr.Row(elem_classes="compact-row"):
                split_duration = gr.Radio(choices=[30, 60], value=30, label="Segment Duration (seconds)", info="30s = fastest training, 60s = more context")
                split_btn = gr.Button("✂️ Split Audio Files", variant="primary", scale=1)

            split_status = gr.Textbox(label="Split Status", interactive=False)

        # ==================== CAPTION ====================
        gr.HTML('<div class="section-title">🏷️ Caption Audio Files</div>')

        gr.HTML("""
        <div class="info-box">
            JSON captions are saved in the <b>same folder</b> as the audio files,
            so the LoRA Training UI will automatically pick them up when you scan the folder.
        </div>
        """)

        with gr.Row(elem_classes="compact-row"):
            caption_input_dir = gr.Textbox(label="Audio Folder", placeholder="Folder with audio files to caption...", scale=3)
            caption_input_picker = gr.Button("📁", scale=0, min_width=45)

        with gr.Row(elem_classes="compact-row"):
            activation_tag_input = gr.Textbox(
                label="🏷️ Activation Tag (required)",
                placeholder="e.g., LNKPRK_style",
                scale=2,
                info="Unique trigger word for LoRA. Saved in every JSON.",
            )
            max_tokens_input = gr.Number(label="Max Tokens", value=512, precision=0, scale=1)
            generate_lyrics_check = gr.Checkbox(label="🎤 Also generate Lyrics", value=True, scale=1, info="ACE-Step Transcriber (structured lyrics with section tags)")
            save_csv_check = gr.Checkbox(label="Also save CSV", value=False, scale=0)
            csv_path_input = gr.Textbox(label="CSV Path (optional)", placeholder="./captions.csv", scale=2, visible=True)

        with gr.Row():
            caption_btn = gr.Button("🏷️ Start Captioning", variant="primary", size="lg", elem_classes="primary-action", scale=2)
            transcribe_only_btn = gr.Button("🎤 Transcribe Lyrics Only", variant="secondary", size="lg", scale=1)

        gr.HTML("""
        <div class="quick-tip">
            <strong>💡 VRAM-saving workflow:</strong> If both models don't fit in VRAM,
            run captioning first (without lyrics), then unload Captioner, load Transcriber,
            and click "Transcribe Lyrics Only" to add structured lyrics to your existing JSON files.
        </div>
        """)

        caption_status = gr.Textbox(label="Progress", interactive=False, lines=2)

        caption_table = gr.Dataframe(
            headers=["#", "Filename", "Caption Preview", "Metadata"],
            datatype=["number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            max_height=300,
        )

        # ==================== PREVIEW & EDIT ====================
        with gr.Accordion("👀 Preview & Edit Captions", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    sample_slider = gr.Slider(0, 0, 0, step=1, label="Sample #")
                    preview_audio = gr.Audio(label="Preview", type="filepath", interactive=False)
                    preview_filename = gr.Textbox(label="Filename", interactive=False)

                with gr.Column(scale=2):
                    edit_caption = gr.Textbox(label="Caption", lines=3)

                    with gr.Row(elem_classes="compact-row"):
                        edit_bpm = gr.Number(label="BPM", precision=0, scale=1)
                        edit_key = gr.Textbox(label="Key", scale=1)
                        edit_timesig = gr.Textbox(label="Time Sig", scale=1)
                        edit_duration = gr.Textbox(label="Duration (s)", interactive=False, scale=1)
                        edit_genre = gr.Textbox(label="Genre", scale=1)
                        edit_language = gr.Textbox(label="Language", scale=1)

                    edit_lyrics = gr.Textbox(label="Lyrics", lines=5, placeholder="[Verse]\nLyrics here...\n\n[Chorus]\n...")
                    with gr.Row():
                        save_edit_btn = gr.Button("💾 Save Edit", variant="secondary")
                        edit_status = gr.Textbox(label="Status", interactive=False, scale=2)

        # ==================== EVENT HANDLERS ====================

        # Folder pickers
        model_path_picker.click(fn=open_folder_picker, outputs=[model_path_input])
        transcriber_path_picker.click(fn=open_folder_picker, outputs=[transcriber_path_input])
        split_input_picker.click(fn=open_folder_picker, outputs=[split_input_dir])
        split_output_picker.click(fn=open_folder_picker, outputs=[split_output_dir])
        caption_input_picker.click(fn=open_folder_picker, outputs=[caption_input_dir])

        # Captioner Model
        load_btn.click(fn=load_model_ui, inputs=[model_path_input], outputs=[model_status])
        unload_btn.click(fn=unload_model_ui, outputs=[model_status])

        # Transcriber Model
        load_transcriber_btn.click(fn=load_transcriber_ui, inputs=[transcriber_path_input, model_path_input], outputs=[transcriber_status])
        unload_transcriber_btn.click(fn=unload_transcriber_ui, outputs=[transcriber_status])

        # Split
        split_btn.click(
            fn=split_audio_files_ui,
            inputs=[split_input_dir, split_output_dir, split_duration],
            outputs=[split_status],
        )

        # Captioning
        caption_btn.click(
            fn=start_captioning_ui,
            inputs=[caption_input_dir, activation_tag_input, generate_lyrics_check, save_csv_check, csv_path_input, max_tokens_input],
            outputs=[caption_status, caption_table, sample_slider],
        )

        # Transcribe Only (add lyrics to existing JSONs)
        transcribe_only_btn.click(
            fn=transcribe_only_ui,
            inputs=[caption_input_dir, activation_tag_input],
            outputs=[caption_status, caption_table, sample_slider],
        )

        # Preview & Edit
        sample_slider.change(
            fn=get_caption_preview,
            inputs=[sample_slider],
            outputs=[preview_audio, preview_filename, edit_caption, edit_lyrics,
                     edit_bpm, edit_key, edit_timesig, edit_duration, edit_genre, edit_language],
        )
        save_edit_btn.click(
            fn=save_caption_edit,
            inputs=[sample_slider, edit_caption, edit_lyrics,
                    edit_bpm, edit_key, edit_timesig, edit_genre, edit_language],
            outputs=[edit_status],
        )

    return demo


# ============== CLI Mode ==============

def main_cli():
    """Original CLI mode."""
    parser = argparse.ArgumentParser(
        description="ACE-Step Audio Captioner - Generate detailed music descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Gradio UI
  python captioner_standalone.py --ui

  # CLI: Generate captions and save to JSON files
  python captioner_standalone.py --input_dir ./my_music --output_dir ./captions

  # CLI: Generate captions and save to CSV
  python captioner_standalone.py --input_dir ./my_music --output_csv ./captions.csv
        """
    )

    parser.add_argument("--ui", action="store_true", help="Launch Gradio web interface")
    parser.add_argument("--port", type=int, default=7862, help="Port for Gradio UI (default: 7862)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Gradio UI")
    parser.add_argument("--input_dir", help="Directory containing audio files to caption")
    parser.add_argument("--output_dir", help="Directory to save individual JSON files (one per audio)")
    parser.add_argument("--output_csv", help="CSV file to save all captions")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path to ACE-Step Captioner model",
    )
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens per caption (default: 512)")

    args = parser.parse_args()

    # UI mode
    if args.ui:
        import gradio as gr
        print(f"🎙️ Starting ACE-Step Captioner UI on {args.host}:{args.port}")
        demo = create_captioner_ui()
        demo.queue()
        allowed_paths = ["/", "C:\\", "D:\\", "E:\\", "F:\\", "G:\\", "H:\\"]
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            show_error=True,
            allowed_paths=allowed_paths,
        )
        return

    # CLI mode - require input_dir
    if not args.input_dir:
        parser.error("the following arguments are required: --input_dir (or use --ui for web interface)")

    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)

    is_hf_id = "/" in args.model_path and "\\" not in args.model_path and not os.path.isabs(args.model_path)
    if not is_hf_id and not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        print("To auto-download from HuggingFace, use format: ACE-Step/acestep-captioner")
        sys.exit(1)

    if not args.output_dir and not args.output_csv:
        print("⚠️ Warning: No output specified. Captions will only be printed to console.")
        print("   Use --output_dir and/or --output_csv to save results.")

    def show_progress(current, total, filename):
        print(f"\n[{current}/{total}] Processing: {filename}")

    def show_load_progress(step, message):
        print(f"  [{step}/4] {message}")

    print("=" * 60)
    print("🎵 ACE-Step Audio Captioner")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_dir}")
    if args.output_dir:
        print(f"Output JSON: {args.output_dir}")
    if args.output_csv:
        print(f"Output CSV: {args.output_csv}")
    print("=" * 60)

    captioner = AceStepCaptioner(args.model_path)

    print("\n🔧 Loading ACE-Step Captioner model...")
    status = captioner.load_model(show_load_progress)
    print(status)

    if "❌" in status:
        sys.exit(1)

    print(f"\n📂 Processing audio files from: {args.input_dir}")
    results = captioner.process_directory(
        args.input_dir,
        args.output_dir,
        args.output_csv,
        show_progress,
        args.max_tokens,
    )

    print("\n" + "=" * 60)
    print(f"✅ Done! Processed {len(results)} files")
    if args.output_dir:
        print(f"   JSON files saved to: {args.output_dir}")
    if args.output_csv:
        print(f"   CSV saved to: {args.output_csv}")
    print("=" * 60)

    captioner.unload_model()


if __name__ == "__main__":
    main_cli()
