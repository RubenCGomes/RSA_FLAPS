from __future__ import annotations

import numpy as np

from ml_app.utils.audio import istft_from_mag_phase, stft_mag_phase


def test_stft_istft_roundtrip_smoke():
    sr = 16000
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    wave = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    magnitude, phase = stft_mag_phase(wave, n_fft=512, hop_length=128)
    reconstructed = istft_from_mag_phase(magnitude, phase, n_fft=512, hop_length=128, length=wave.shape[-1])

    assert reconstructed.shape == wave.shape
    assert np.mean(np.abs(reconstructed - wave)) < 1e-2

