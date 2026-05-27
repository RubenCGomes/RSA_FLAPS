from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..utils.audio import (
    EPS,
    load_audio_segment,
    normalize_magnitude,
    pad_or_trim,
    stft_mag_phase,
    stem_names,
)


@dataclass(frozen=True)
class SegmentRecord:
    split: str
    track_name: str
    track_dir: str
    segment_index: int
    start_sample: int
    num_samples: int | None
    sample_rate: int
    files: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "split": self.split,
            "track_name": self.track_name,
            "track_dir": self.track_dir,
            "segment_index": self.segment_index,
            "start_sample": self.start_sample,
            "num_samples": self.num_samples if self.num_samples is not None else None,
            "sample_rate": self.sample_rate,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SegmentRecord":
        return cls(
            split=str(payload["split"]),
            track_name=str(payload["track_name"]),
            track_dir=str(payload["track_dir"]),
            segment_index=int(payload["segment_index"]),
            start_sample=int(payload["start_sample"]),
            num_samples=None if payload.get("num_samples") is None else int(payload["num_samples"]),
            sample_rate=int(payload["sample_rate"]),
            files={str(k): str(v) for k, v in dict(payload["files"]).items()},
        )


class MusdbManifestBuilder:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        target_sr: int = 16_000,
        segment_seconds: float | None = 5.0,
        hop_seconds: float | None = None,
        mono: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
    ):
        self.root = Path(root)
        self.target_sr = int(target_sr)
        # segment_seconds == None signals "use whole tracks"
        self.segment_seconds = None if segment_seconds is None else float(segment_seconds)
        if hop_seconds is not None:
            self.hop_seconds = float(hop_seconds)
        else:
            self.hop_seconds = None if self.segment_seconds is None else float(self.segment_seconds)
        self.mono = mono
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, split: str) -> Path:
        seg_suffix = "full" if self.segment_seconds is None else f"{self.segment_seconds:g}"
        hop_suffix = "full" if self.hop_seconds is None else f"{self.hop_seconds:g}"
        suffix = f"sr{self.target_sr}_seg{seg_suffix}_hop{hop_suffix}_{'mono' if self.mono else 'stereo'}_v2"
        return self.cache_dir / f"manifest_{split}_{suffix}.json"

    def build(self, split: str) -> List[SegmentRecord]:
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected MUSDB split directory: {split_dir}")

        records: List[SegmentRecord] = []
        for track_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            files = self._track_files(track_dir)
            if not files:
                continue

            with sf.SoundFile(files["mixture"]) as handle:
                total_frames = handle.frames
                source_sr = handle.samplerate

            # If segment_seconds is None -> treat whole track as a single record
            if self.segment_seconds is None:
                records.append(
                    SegmentRecord(
                        split=split,
                        track_name=track_dir.name,
                        track_dir=str(track_dir),
                        segment_index=0,
                        start_sample=0,
                        num_samples=None,  # signal loader to read full file
                        sample_rate=self.target_sr,
                        files=files,
                    )
                )
            else:
                # start_sample and num_samples are stored in SOURCE sample-rate
                # frames so that sf.SoundFile.seek() / .read() land at the correct
                # position.  load_audio_segment scales them to target_sr after
                # resampling.
                source_segment_frames = max(1, int(round(source_sr * self.segment_seconds)))
                source_hop_frames = max(1, int(round(source_sr * self.hop_seconds)))
                estimated_segments = max(1, int(np.ceil(total_frames / source_hop_frames)))

                for segment_index in range(estimated_segments):
                    start_sample = segment_index * source_hop_frames
                    records.append(
                        SegmentRecord(
                            split=split,
                            track_name=track_dir.name,
                            track_dir=str(track_dir),
                            segment_index=segment_index,
                            start_sample=start_sample,
                            num_samples=source_segment_frames,
                            sample_rate=self.target_sr,
                            files=files,
                        )
                    )
        return records

    def load(self, split: str, force_rebuild: bool = False) -> List[SegmentRecord]:
        manifest_path = self.manifest_path(split)
        if manifest_path.exists() and not force_rebuild:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return [SegmentRecord.from_dict(item) for item in payload]

        records = self.build(split)
        # Write atomically via a temp file + rename so concurrent clients never
        # read a partially-written manifest.
        tmp_path = manifest_path.with_suffix(".json.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump([record.to_dict() for record in records], handle, indent=2)
            tmp_path.replace(manifest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return records

    @staticmethod
    def _track_files(track_dir: Path) -> Dict[str, str]:
        files = {name: str(track_dir / f"{name}.wav") for name in stem_names()}
        files["mixture"] = str(track_dir / "mixture.wav")
        if not Path(files["mixture"]).exists():
            return {}
        return {key: value for key, value in files.items() if Path(value).exists()}


class MusdbSeparationDataset(Dataset):
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        split: str = "train",
        target_sr: int = 16_000,
        segment_seconds: float | None = 5.0,
        hop_seconds: float | None = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        mono: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        max_tracks: int | None = None,
        max_segments: int | None = None,
        client_id: int | None = None,
        num_clients: int | None = None,
        eval_fraction: float = 0.15,
        augment_train: bool = False,
        augment_gain_db: float = 6.0,
        augment_noise_std: float = 0.0,
        augment_stem_dropout_prob: float = 0.0,
        augment_shift_seconds: float = 0.0,
        augment_remix_prob: float = 0.0,
        augment_phase_flip: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.target_sr = int(target_sr)
        # allow None for full-track usage
        self.segment_seconds = None if segment_seconds is None else float(segment_seconds)
        if hop_seconds is not None:
            self.hop_seconds = float(hop_seconds)
        else:
            self.hop_seconds = None if self.segment_seconds is None else float(self.segment_seconds)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.mono = mono
        self.max_tracks = max_tracks
        self.max_segments = max_segments
        self.client_id = client_id
        self.num_clients = num_clients
        self.eval_fraction = float(eval_fraction)
        self.augment_train = bool(augment_train)
        self.augment_gain_db = max(0.0, float(augment_gain_db))
        self.augment_noise_std = max(0.0, float(augment_noise_std))
        self.augment_stem_dropout_prob = min(1.0, max(0.0, float(augment_stem_dropout_prob)))
        self.augment_shift_seconds = max(0.0, float(augment_shift_seconds))
        self.augment_remix_prob = min(1.0, max(0.0, float(augment_remix_prob)))
        self.augment_phase_flip = bool(augment_phase_flip)
        self.builder = MusdbManifestBuilder(
            self.root,
            target_sr=self.target_sr,
            segment_seconds=self.segment_seconds,
            hop_seconds=self.hop_seconds,
            mono=self.mono,
            cache_dir=cache_dir,
        )
        self.records = self._select_records()

    @staticmethod
    def _track_hash(track_name: str) -> int:
        digest = hashlib.sha1(track_name.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _select_records(self) -> List[SegmentRecord]:
        base_split = "train" if self.split in {"train", "val"} else "test"
        records = self.builder.load(base_split)

        if base_split == "train":
            threshold = int(self.eval_fraction * 100)
            if self.split == "val":
                records = [record for record in records if self._track_hash(record.track_name) % 100 < threshold]
            elif self.split == "train":
                records = [record for record in records if self._track_hash(record.track_name) % 100 >= threshold]

        if self.max_tracks is not None:
            selected_tracks: List[str] = []
            filtered: List[SegmentRecord] = []
            for record in records:
                if record.track_name not in selected_tracks:
                    if len(selected_tracks) >= self.max_tracks:
                        continue
                    selected_tracks.append(record.track_name)
                filtered.append(record)
            records = filtered

        if self.num_clients is not None and self.client_id is not None and self.num_clients > 0:
            records = [record for record in records if self._track_hash(record.track_name) % self.num_clients == self.client_id]

        if self.max_segments is not None:
            records = records[: self.max_segments]

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        rng = np.random.default_rng()
        
        # Resolve paths relative to current root to support moved datasets
        def resolve_path(original_path: str) -> str:
            path_obj = Path(original_path)
            # Find the split directory (train/test) in the path to get the relative part
            parts = path_obj.parts
            if "train" in parts:
                split_idx = parts.index("train")
            elif "test" in parts:
                split_idx = parts.index("test")
            else:
                return original_path # Fallback
            
            relative_path = Path(*parts[split_idx:])
            return str(self.root / relative_path)

        source_waves: Dict[str, np.ndarray] = {}
        source_mags: List[np.ndarray] = []
        
        # Determine actual sources (possibly remixed)
        active_records = {stem: record for stem in stem_names()}
        if self.split == "train" and self.augment_train and self.augment_remix_prob > 0:
            for stem in stem_names():
                if rng.random() < self.augment_remix_prob:
                    # Pick a random segment from a potentially different track
                    active_records[stem] = self.records[rng.integers(0, len(self.records))]

        # Load sources
        expected_len: int | None = None if self.segment_seconds is None else int(round(self.target_sr * self.segment_seconds))
        for stem, rec in active_records.items():
            if stem in rec.files:
                source, _ = load_audio_segment(
                    resolve_path(rec.files[stem]),
                    start_sample=rec.start_sample,
                    num_samples=rec.num_samples,
                    target_sr=self.target_sr,
                    mono=self.mono,
                )
            else:
                # placeholder, will fill zeros after we know expected_len
                source = np.array([], dtype=np.float32)
            source_waves[stem] = source

        # If full-track mode, determine expected length from longest available stem
        if expected_len is None:
            lengths = [wave.shape[0] for wave in source_waves.values() if wave is not None]
            expected_len = int(max(lengths) if lengths else 0)

        # Pad or trim all sources to expected length (fills missing stems with zeros)
        for stem in list(source_waves.keys()):
            wave = source_waves[stem]
            if wave is None or wave.size == 0:
                wave = np.zeros((expected_len,), dtype=np.float32)
            source_waves[stem] = pad_or_trim(wave, expected_len)

        if self.split == "train" and self.augment_train:
            source_waves, mixture_wave = self._augment_sources(source_waves, expected_len)
        else:
            mixture_wave = np.sum(np.stack(list(source_waves.values()), axis=0), axis=0).astype(np.float32)

        # Final normalization to ensure no clipping
        peak = np.max(np.abs(mixture_wave))
        if peak > 1.0:
            mixture_wave /= (peak + 1e-6)
            for stem in stem_names():
                source_waves[stem] /= (peak + 1e-6)

        mix_mag_raw, mix_phase = stft_mag_phase(mixture_wave, n_fft=self.n_fft, hop_length=self.hop_length)
        
        for stem in stem_names():
            source_mag, _ = stft_mag_phase(source_waves[stem], n_fft=self.n_fft, hop_length=self.hop_length)
            source_mags.append(source_mag)

        source_stack = np.stack(source_mags, axis=0)
        source_power = source_stack**2
        mask_targets = source_power / (np.sum(source_power, axis=0, keepdims=True) + EPS)
        mix_mag_input = normalize_magnitude(mix_mag_raw, log_scale=True)

        return {
            "mix_mag": torch.from_numpy(mix_mag_input[None, ...].astype(np.float32)),
            "mix_mag_raw": torch.from_numpy(mix_mag_raw.astype(np.float32)),
            "mix_phase": torch.from_numpy(mix_phase.astype(np.float32)),
            "target_masks": torch.from_numpy(mask_targets.astype(np.float32)),
            "target_mags": torch.from_numpy(source_stack.astype(np.float32)),
            "mixture_wave": torch.from_numpy(mixture_wave.astype(np.float32)),
            "source_waves": {stem: torch.from_numpy(wave.astype(np.float32)) for stem, wave in source_waves.items()},
            "record": record,
            "sample_rate": self.target_sr,
        }

    def summary(self) -> Dict[str, object]:
        track_names = sorted({record.track_name for record in self.records})
        return {
            "split": self.split,
            "tracks": len(track_names),
            "segments": len(self.records),
            "sample_rate": self.target_sr,
            "segment_seconds": self.segment_seconds,
            "hop_seconds": self.hop_seconds,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "augment_train": self.augment_train,
        }

    def track_names(self) -> set[str]:
        return {record.track_name for record in self.records}

    def preview(self, limit: int = 3) -> List[Dict[str, object]]:
        return [
            {
                "track_name": record.track_name,
                "segment_index": record.segment_index,
                "start_sample": record.start_sample,
                "files": record.files,
            }
            for record in self.records[:limit]
        ]

    def _augment_sources(
        self,
        source_waves: Dict[str, np.ndarray],
        expected_len: int,
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        rng = np.random.default_rng()
        augmented: Dict[str, np.ndarray] = {}
        active_stems = 0
        max_shift = int(round(self.augment_shift_seconds * self.target_sr))

        for stem in stem_names():
            wave = np.array(source_waves[stem], dtype=np.float32, copy=True)
            if wave.shape[0] != expected_len:
                wave = pad_or_trim(wave, expected_len)

            if self.augment_gain_db > 0:
                gain_db = rng.uniform(-self.augment_gain_db, self.augment_gain_db)
                wave *= np.float32(10.0 ** (gain_db / 20.0))

            if self.augment_phase_flip and rng.random() < 0.5:
                wave *= -1.0

            dropped = self.augment_stem_dropout_prob > 0 and rng.random() < self.augment_stem_dropout_prob
            if dropped:
                wave.fill(0.0)
            elif max_shift > 0:
                shift = int(rng.integers(-max_shift, max_shift + 1))
                if shift != 0:
                    wave = np.roll(wave, shift)

            if np.any(np.abs(wave) > 1e-7):
                active_stems += 1
            augmented[stem] = wave.astype(np.float32, copy=False)

        if active_stems == 0:
            keep_stem = stem_names()[int(rng.integers(0, len(stem_names())))]
            augmented[keep_stem] = np.array(source_waves[keep_stem], dtype=np.float32, copy=True)

        mixture_wave = np.sum(np.stack([augmented[stem] for stem in stem_names()], axis=0), axis=0).astype(np.float32)

        if self.augment_noise_std > 0:
            noise = rng.normal(0.0, self.augment_noise_std, size=mixture_wave.shape).astype(np.float32)
            mixture_wave = mixture_wave + noise

        peak = max(
            float(np.max(np.abs(mixture_wave))),
            max(float(np.max(np.abs(wave))) for wave in augmented.values()),
        )
        if peak > 1.0:
            scale = np.float32(0.98 / peak)
            mixture_wave *= scale
            for stem in stem_names():
                augmented[stem] = augmented[stem] * scale

        return augmented, mixture_wave

