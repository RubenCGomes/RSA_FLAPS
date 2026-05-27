from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Tuple

import librosa
import numpy as np
import soundfile as sf

EPS = 1e-8


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return np.mean(audio, axis=1).astype(np.float32, copy=False)


@lru_cache(maxsize=512)
def _resample_key(sr_in: int, sr_out: int) -> Tuple[int, int]:
    return sr_in, sr_out


def resample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    return librosa.resample(audio.astype(np.float32, copy=False), orig_sr=sr_in, target_sr=sr_out)


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return audio.astype(np.float32, copy=False)
    if audio.shape[0] == length:
        return audio.astype(np.float32, copy=False)
    if audio.shape[0] > length:
        return audio[:length].astype(np.float32, copy=False)
    out = np.zeros(length, dtype=np.float32)
    out[: audio.shape[0]] = audio.astype(np.float32, copy=False)
    return out


def load_audio_segment(
    path: str,
    *,
    start_sample: int = 0,
    num_samples: int | None = None,
    target_sr: int | None = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    # num_samples and start_sample are in the file's NATIVE sample rate.
    # We compute the expected output length in target_sr units before resampling
    # so the final pad/trim lands at the right number of frames.
    with sf.SoundFile(path) as handle:
        source_sr = handle.samplerate
        if start_sample:
            handle.seek(max(0, start_sample))
        frames = -1 if num_samples is None else max(0, num_samples)
        audio = handle.read(frames=frames, dtype="float32", always_2d=False)

    # Compute expected output length in the output sample rate
    if num_samples is not None and target_sr is not None and target_sr != source_sr:
        expected_samples = int(round(num_samples * target_sr / source_sr))
    else:
        expected_samples = num_samples

    if mono:
        audio = ensure_mono(audio)
    else:
        audio = audio.astype(np.float32, copy=False)

    if target_sr is not None and target_sr != source_sr:
        if audio.ndim == 1:
            audio = resample_audio(audio, source_sr, target_sr)
        else:
            channels = [resample_audio(audio[:, ch], source_sr, target_sr) for ch in range(audio.shape[1])]
            max_len = max((ch.shape[0] for ch in channels), default=0)
            audio = np.stack([pad_or_trim(ch, max_len) for ch in channels], axis=1)

    if expected_samples is not None:
        if audio.ndim == 1:
            audio = pad_or_trim(audio, expected_samples)
        else:
            audio = np.stack([pad_or_trim(audio[:, ch], expected_samples) for ch in range(audio.shape[1])], axis=1)

    return audio.astype(np.float32, copy=False), target_sr if target_sr is not None else source_sr


def stft_mag_phase(
    audio: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if audio.ndim != 1:
        audio = ensure_mono(audio)
    complex_spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude = np.abs(complex_spec).astype(np.float32)
    phase = np.angle(complex_spec).astype(np.float32)
    return magnitude, phase


def istft_from_mag_phase(
    magnitude: np.ndarray,
    phase: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    length: int | None = None,
) -> np.ndarray:
    complex_spec = magnitude * np.exp(1j * phase)
    audio = librosa.istft(complex_spec, hop_length=hop_length, win_length=None, length=length)
    return audio.astype(np.float32, copy=False)


def normalize_magnitude(magnitude: np.ndarray, log_scale: bool = True) -> np.ndarray:
    magnitude = magnitude.astype(np.float32, copy=False)
    if log_scale:
        magnitude = np.log1p(magnitude)
    max_value = float(np.max(magnitude)) if magnitude.size else 1.0
    if max_value <= EPS:
        return magnitude
    return magnitude / max_value


def denormalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(magnitude, 0.0, None)).astype(np.float32, copy=False)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = ensure_mono(reference)
    estimate = ensure_mono(estimate)
    min_len = min(reference.shape[0], estimate.shape[0])
    if min_len == 0:
        return 0.0
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    ref_energy = np.sum(reference ** 2) + EPS
    scale = np.sum(reference * estimate) / ref_energy
    projection = scale * reference
    noise = estimate - projection
    ratio = (np.sum(projection ** 2) + EPS) / (np.sum(noise ** 2) + EPS)
    return float(10.0 * np.log10(ratio + EPS))


def stem_names() -> Tuple[str, ...]:
    return ("vocals", "drums", "bass", "other")


def stack_stems(stems: Dict[str, np.ndarray], order: Iterable[str] | None = None) -> np.ndarray:
    ordered = tuple(order) if order is not None else stem_names()
    return np.stack([stems[name].astype(np.float32, copy=False) for name in ordered], axis=0)

