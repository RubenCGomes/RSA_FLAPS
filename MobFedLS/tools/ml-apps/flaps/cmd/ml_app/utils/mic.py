from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .audio import ensure_mono, pad_or_trim


def record_microphone(
    duration_seconds: float,
    *,
    sample_rate: int = 44_100,
    channels: int = 1,
    device: str | int | None = None,
    dtype: str = "float32",
    wait: bool = True,
) -> np.ndarray:
    """Record audio from the default microphone and return a mono float32 waveform."""
    try:
        sd = importlib.import_module("sounddevice")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sounddevice is required for microphone recording. Install it with 'pip install sounddevice' and make sure PortAudio is available on Linux."
        ) from exc

    frames = max(1, int(round(duration_seconds * sample_rate)))
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        device=device,
        dtype=dtype,
    )
    if wait:
        sd.wait()
    audio = np.asarray(recording, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = ensure_mono(audio)
    elif audio.ndim == 2:
        audio = audio[:, 0]
    return pad_or_trim(audio, frames)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(str(path), audio.astype(np.float32, copy=False), sample_rate)


