from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest
import torch

from ml_app.models.unet_small import UNetSmall
from ml_app.utils.losses import stem_classification_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(sr: int = 8000, duration: float = 1.0) -> bytes:
    """Return a minimal WAV file with a sine wave."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        pcm = (samples * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Classification head output
# ---------------------------------------------------------------------------

def test_cls_logits_are_unbounded():
    """cls_logits should be raw logits (not sigmoidised)."""
    model = UNetSmall(in_channels=1, out_channels=4, base_filters=8)
    x = torch.randn(1, 1, 65, 65)
    _, cls_logits = model(x)
    # Raw logits can be outside [0, 1]; if they were all in [0,1] the head
    # would be accidentally applying sigmoid, which is a bug.
    assert cls_logits.shape == (1, 4)


def test_stem_classification_loss_zero_on_perfect_prediction():
    """BCE loss should be near zero when logits strongly match labels."""
    batch = 4
    n_stems = 4
    stem_idx = 2  # vocals

    # Large positive logit → high confidence present
    logits = torch.zeros(batch, n_stems)
    logits[:, stem_idx] = 10.0
    labels = torch.ones(batch, n_stems)

    loss = stem_classification_loss(logits, labels, stem_idx)
    assert loss.item() < 0.01


def test_stem_classification_loss_high_on_wrong_prediction():
    logits = torch.zeros(4, 4)
    logits[:, 1] = -10.0  # predicts absent
    labels = torch.ones(4, 4)  # actually present

    loss = stem_classification_loss(logits, labels, stem_index=1)
    assert loss.item() > 5.0


def test_cls_loss_only_uses_assigned_stem_column():
    """Changing logits for non-target stems must not affect the loss."""
    logits_a = torch.zeros(2, 4)
    logits_b = logits_a.clone()
    logits_b[:, 0] = 99.0  # unrelated stem
    logits_b[:, 3] = -99.0

    labels = torch.ones(2, 4)
    loss_a = stem_classification_loss(logits_a, labels, stem_index=2)
    loss_b = stem_classification_loss(logits_b, labels, stem_index=2)
    assert torch.isclose(loss_a, loss_b)


# ---------------------------------------------------------------------------
# MLApp.classify()
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_app(tmp_path):
    from ml_app.mfls_app import MLApp

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    return MLApp(
        data_root=str(dataset),
        sr=8000,
        n_fft=512,
        hop_length=128,
        base_filters=8,
        device="cpu",
        use_amp=False,
        target_stem="vocals",
    )


def test_classify_returns_assigned_stem(tiny_app):
    wav = _make_wav_bytes(sr=8000)
    result = tiny_app.classify(wav)
    assert result["stem"] == "vocals"
    assert isinstance(result["present"], bool)
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_without_target_stem_returns_all_stems(tmp_path):
    from ml_app.mfls_app import MLApp

    app = MLApp(
        data_root=str(tmp_path / "dataset"),
        sr=8000,
        n_fft=512,
        hop_length=128,
        base_filters=8,
        device="cpu",
        use_amp=False,
        target_stem=None,
    )
    wav = _make_wav_bytes(sr=8000)
    result = app.classify(wav)
    assert "stems" in result
    assert set(result["stems"].keys()) == {"drums", "bass", "vocals", "other"}
    for info in result["stems"].values():
        assert isinstance(info["present"], bool)
        assert 0.0 <= info["confidence"] <= 1.0


def test_classify_is_deterministic(tiny_app):
    wav = _make_wav_bytes(sr=8000)
    r1 = tiny_app.classify(wav)
    r2 = tiny_app.classify(wav)
    assert r1["confidence"] == pytest.approx(r2["confidence"])