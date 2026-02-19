"""
PyTorch Lightning DataModule for LoRA Training

Handles data loading and preprocessing for training ACE-Step LoRA adapters.
Supports both raw audio loading and preprocessed tensor loading.
"""

import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

try:
    from lightning.pytorch import LightningDataModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning not installed. Training module will not be available.")
    # Create a dummy class for type hints
    class LightningDataModule:
        pass


# ============================================================================
# Preprocessed Tensor Dataset (Recommended for Training)
# ============================================================================

class PreprocessedTensorDataset(Dataset):
    """Dataset that loads preprocessed tensor files.

    This is the recommended dataset for training as all tensors are pre-computed:
    - target_latents: VAE-encoded audio [T, 64]
    - encoder_hidden_states: Condition encoder output [L, D]
    - encoder_attention_mask: Condition mask [L]
    - context_latents: Source context [T, 65]
    - attention_mask: Audio latent mask [T]

    No VAE/text encoder needed during training - just load tensors directly!
    All samples are cached in RAM on first load to avoid repeated disk I/O.

    Supports optional random cropping of long sequences to reduce compute cost.
    The attention cost in DiT is O(T²), so cropping 240s audio (T=6000) to 60s
    (T=1500) gives a ~16× speedup per step. Each epoch sees a different random
    window, providing data augmentation.
    """

    def __init__(self, tensor_dir: str, verify_checksums: bool = False,
                 max_latent_length: int = 0):
        """Initialize from a directory of preprocessed .pt files.

        Args:
            tensor_dir: Directory containing preprocessed .pt files and manifest.json
            verify_checksums: If True and manifest has MD5 checksums, verify file integrity
            max_latent_length: Max latent time-steps per sample (0 = no crop).
                Set to e.g. 1500 (~60s) to match inference length and greatly
                reduce training time.  A random window is chosen each __getitem__
                call, providing implicit data augmentation.
        """
        self.tensor_dir = tensor_dir
        self.max_latent_length = max_latent_length
        self.sample_paths = []
        self.manifest = None
        self._checksum_map = {}  # path -> expected md5

        # Load manifest if exists
        manifest_path = os.path.join(tensor_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            samples = self.manifest.get("samples", [])
            self.sample_paths = []
            for s in samples:
                p = s.get("path") if isinstance(s, dict) else s
                if not isinstance(p, str):
                    continue
                if not os.path.isabs(p):
                    p = os.path.join(tensor_dir, p)
                self.sample_paths.append(os.path.normpath(p))

            # Build checksum map from v2 manifest
            for detail in self.manifest.get("sample_details", []):
                if "md5" in detail and "path" in detail:
                    p = detail["path"]
                    if not os.path.isabs(p):
                        p = os.path.join(tensor_dir, p)
                    self._checksum_map[os.path.normpath(p)] = detail["md5"]

            manifest_ver = self.manifest.get("version", 1)
            num_samples = self.manifest.get("num_samples", len(self.sample_paths))
            logger.info(f"Loaded manifest v{manifest_ver}: {num_samples} samples")
        else:
            # Fallback: scan directory for .pt files
            for f in os.listdir(tensor_dir):
                if f.endswith('.pt') and f != "manifest.json":
                    self.sample_paths.append(os.path.join(tensor_dir, f))
            logger.warning(f"No manifest.json found, discovered {len(self.sample_paths)} .pt files")

        # Validate paths
        self.valid_paths = [p for p in self.sample_paths if os.path.exists(p)]

        if len(self.valid_paths) != len(self.sample_paths):
            missing = len(self.sample_paths) - len(self.valid_paths)
            logger.warning(f"Some tensor files not found: {missing} missing")

        # Verify checksums if requested and available
        if verify_checksums and self._checksum_map:
            import hashlib
            corrupted = []
            for p in self.valid_paths:
                expected = self._checksum_map.get(p)
                if expected:
                    with open(p, "rb") as bf:
                        actual = hashlib.md5(bf.read()).hexdigest()
                    if actual != expected:
                        corrupted.append(p)
                        logger.error(f"Checksum mismatch: {p} (expected {expected}, got {actual})")
            if corrupted:
                logger.warning(f"{len(corrupted)} files failed checksum verification, excluding them")
                self.valid_paths = [p for p in self.valid_paths if p not in corrupted]

        # Cache all samples in RAM (preprocessed tensors are small, ~1-2MB each)
        self._cache = []
        for p in self.valid_paths:
            data = torch.load(p, map_location='cpu')
            self._cache.append({
                "target_latents": data["target_latents"],
                "attention_mask": data["attention_mask"],
                "encoder_hidden_states": data["encoder_hidden_states"],
                "encoder_attention_mask": data["encoder_attention_mask"],
                "context_latents": data["context_latents"],
                "metadata": data.get("metadata", {}),
            })

        logger.info(f"PreprocessedTensorDataset: {len(self._cache)} samples cached in RAM from {tensor_dir}")

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a cached preprocessed sample, optionally random-cropped.

        When max_latent_length > 0, a random contiguous window of that length
        is extracted from the time dimension of target_latents, attention_mask
        and context_latents.  This keeps compute bounded regardless of the
        original audio duration, and each epoch sees a different window
        (implicit data augmentation).

        Returns:
            Dictionary containing all pre-computed tensors for training
        """
        sample = self._cache[idx]

        T = sample["target_latents"].shape[0]  # time dimension

        if self.max_latent_length > 0 and T > self.max_latent_length:
            # Random crop along time axis
            max_start = T - self.max_latent_length
            start = random.randint(0, max_start)
            end = start + self.max_latent_length

            return {
                "target_latents": sample["target_latents"][start:end],      # [T', 64]
                "attention_mask": sample["attention_mask"][start:end],       # [T']
                "context_latents": sample["context_latents"][start:end],     # [T', 128]
                # encoder states are text-conditioned, not time-aligned — keep full
                "encoder_hidden_states": sample["encoder_hidden_states"],
                "encoder_attention_mask": sample["encoder_attention_mask"],
                "metadata": sample["metadata"],
            }

        return sample


def collate_preprocessed_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for preprocessed tensor batches.

    Handles variable-length tensors by padding to the longest in the batch.
    Uses torch.nn.utils.rnn.pad_sequence for efficient native padding.

    Args:
        batch: List of sample dictionaries with pre-computed tensors

    Returns:
        Batched dictionary with all tensors stacked
    """
    from torch.nn.utils.rnn import pad_sequence

    # pad_sequence pads along dim=0 (the time/sequence dimension) and stacks into batch
    target_latents = pad_sequence([s["target_latents"] for s in batch], batch_first=True, padding_value=0.0)
    attention_masks = pad_sequence([s["attention_mask"] for s in batch], batch_first=True, padding_value=0.0)
    context_latents = pad_sequence([s["context_latents"] for s in batch], batch_first=True, padding_value=0.0)
    encoder_hidden_states = pad_sequence([s["encoder_hidden_states"] for s in batch], batch_first=True, padding_value=0.0)
    encoder_attention_masks = pad_sequence([s["encoder_attention_mask"] for s in batch], batch_first=True, padding_value=0.0)

    return {
        "target_latents": target_latents,           # [B, T, 64]
        "attention_mask": attention_masks,           # [B, T]
        "encoder_hidden_states": encoder_hidden_states,  # [B, L, D]
        "encoder_attention_mask": encoder_attention_masks,  # [B, L]
        "context_latents": context_latents,          # [B, T, 65]
        "metadata": [s["metadata"] for s in batch],
    }


class PreprocessedDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for preprocessed tensor files.
    
    This is the recommended DataModule for training. It loads pre-computed tensors
    directly without needing VAE, text encoder, or condition encoder at training time.
    """
    
    def __init__(
        self,
        tensor_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.0,
        max_latent_length: int = 0,
    ):
        """Initialize the data module.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            batch_size: Training batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            val_split: Fraction of data for validation (0 = no validation)
            max_latent_length: Max time-steps per sample (0 = full length).
                1500 ≈ 60s, 3000 ≈ 120s.  Shorter = faster training.
        """
        if LIGHTNING_AVAILABLE:
            super().__init__()

        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.max_latent_length = max_latent_length

        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            # Create full dataset
            full_dataset = PreprocessedTensorDataset(
                self.tensor_dir,
                max_latent_length=self.max_latent_length,
            )
            
            # Split if validation requested
            if self.val_split > 0 and len(full_dataset) > 1:
                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [n_train, n_val]
                )
            else:
                self.train_dataset = full_dataset
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_preprocessed_batch,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
        )


# ============================================================================
# Raw Audio Dataset (Legacy - for backward compatibility)
# ============================================================================

class AceStepTrainingDataset(Dataset):
    """Dataset for ACE-Step LoRA training from raw audio.
    
    DEPRECATED: Use PreprocessedTensorDataset instead for better performance.
    
    Audio Format Requirements (handled automatically):
    - Sample rate: 48kHz (resampled if different)
    - Channels: Stereo (2 channels, mono is duplicated)
    - Max duration: 240 seconds (4 minutes)
    - Min duration: 5 seconds (padded if shorter)
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dit_handler,
        max_duration: float = 240.0,
        target_sample_rate: int = 48000,
    ):
        """Initialize the dataset."""
        self.samples = samples
        self.dit_handler = dit_handler
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        
        self.valid_samples = self._validate_samples()
        logger.info(f"Dataset initialized with {len(self.valid_samples)} valid samples")
    
    def _validate_samples(self) -> List[Dict[str, Any]]:
        """Validate and filter samples."""
        valid = []
        for i, sample in enumerate(self.samples):
            audio_path = sample.get("audio_path", "")
            if not audio_path or not os.path.exists(audio_path):
                logger.warning(f"Sample {i}: Audio file not found: {audio_path}")
                continue
            
            if not sample.get("caption"):
                logger.warning(f"Sample {i}: Missing caption")
                continue
            
            valid.append(sample)
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        sample = self.valid_samples[idx]
        
        audio_path = sample["audio_path"]
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to 48kHz
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        
        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        
        # Truncate/pad
        max_samples = int(self.max_duration * self.target_sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        min_samples = int(5.0 * self.target_sample_rate)
        if audio.shape[1] < min_samples:
            padding = min_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return {
            "audio": audio,
            "caption": sample.get("caption", ""),
            "lyrics": sample.get("lyrics", "[Instrumental]"),
            "metadata": {
                "caption": sample.get("caption", ""),
                "lyrics": sample.get("lyrics", "[Instrumental]"),
                "bpm": sample.get("bpm"),
                "keyscale": sample.get("keyscale", ""),
                "timesignature": sample.get("timesignature", ""),
                "duration": sample.get("duration", audio.shape[1] / self.target_sample_rate),
                "language": sample.get("language", "unknown"),
                "is_instrumental": sample.get("is_instrumental", True),
            },
            "audio_path": audio_path,
        }


def collate_training_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for raw audio batches (legacy)."""
    max_len = max(sample["audio"].shape[1] for sample in batch)
    
    padded_audio = []
    attention_masks = []
    
    for sample in batch:
        audio = sample["audio"]
        audio_len = audio.shape[1]
        
        if audio_len < max_len:
            padding = max_len - audio_len
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        padded_audio.append(audio)
        
        mask = torch.ones(max_len)
        if audio_len < max_len:
            mask[audio_len:] = 0
        attention_masks.append(mask)
    
    return {
        "audio": torch.stack(padded_audio),
        "attention_mask": torch.stack(attention_masks),
        "captions": [s["caption"] for s in batch],
        "lyrics": [s["lyrics"] for s in batch],
        "metadata": [s["metadata"] for s in batch],
        "audio_paths": [s["audio_path"] for s in batch],
    }


class AceStepDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for raw audio loading (legacy).
    
    DEPRECATED: Use PreprocessedDataModule for better training performance.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dit_handler,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_duration: float = 240.0,
        val_split: float = 0.0,
    ):
        if LIGHTNING_AVAILABLE:
            super().__init__()
        
        self.samples = samples
        self.dit_handler = dit_handler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_duration = max_duration
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            if self.val_split > 0 and len(self.samples) > 1:
                n_val = max(1, int(len(self.samples) * self.val_split))
                
                indices = list(range(len(self.samples)))
                random.shuffle(indices)
                
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]
                
                train_samples = [self.samples[i] for i in train_indices]
                val_samples = [self.samples[i] for i in val_indices]
                
                self.train_dataset = AceStepTrainingDataset(
                    train_samples, self.dit_handler, self.max_duration
                )
                self.val_dataset = AceStepTrainingDataset(
                    val_samples, self.dit_handler, self.max_duration
                )
            else:
                self.train_dataset = AceStepTrainingDataset(
                    self.samples, self.dit_handler, self.max_duration
                )
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_training_batch,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_training_batch,
        )


def load_dataset_from_json(json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a dataset from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    samples = data.get("samples", [])
    
    return samples, metadata
