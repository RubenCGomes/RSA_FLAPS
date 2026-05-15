from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ml_app.utils.mic import record_microphone


def test_record_microphone_uses_sounddevice_monkeypatch(monkeypatch):
    frames = 1600
    fake_audio = np.linspace(-1.0, 1.0, frames, dtype=np.float32).reshape(frames, 1)

    fake_sd = types.SimpleNamespace(
        rec=lambda *args, **kwargs: fake_audio,
        wait=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    recorded = record_microphone(0.1, sample_rate=16000, channels=1, wait=True)
    assert recorded.shape == (frames,)
    assert np.allclose(recorded, fake_audio[:, 0])


def test_predict_from_mic_main_uses_recording_and_predict(monkeypatch, tmp_path):
    import scripts.predict_from_mic as mic_script

    captured = {}

    class DummyApp:
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.model = types.SimpleNamespace(load_state_dict=lambda state: None)
            self.run_path = None

        def set_run_path(self, run_path):
            self.run_path = Path(run_path)

        def predict(self, parameters, inputs):
            captured["parameters"] = parameters
            captured["inputs"] = inputs
            return [{"input": "microphone", "stems": {"vocals": "vocals.wav"}}]

    monkeypatch.setattr(mic_script, "MLApp", DummyApp)
    monkeypatch.setattr(mic_script, "load_checkpoint", lambda app, checkpoint_path: None)
    monkeypatch.setattr(mic_script, "record_microphone", lambda *args, **kwargs: np.ones(1600, dtype=np.float32))
    monkeypatch.setattr(mic_script, "save_wav", lambda *args, **kwargs: None)
    monkeypatch.setattr(mic_script, "parse_args", lambda: types.SimpleNamespace(
        seconds=0.1,
        sample_rate=16000,
        channels=1,
        device=None,
        output_dir=str(tmp_path / "mic-runs"),
        checkpoint=str(tmp_path / "missing.pt"),
        data_root=None,
        client_id=None,
        num_clients=None,
    ))

    exit_code = mic_script.main()

    assert exit_code == 0
    assert captured["parameters"] is None
    assert isinstance(captured["inputs"], list)
    assert captured["inputs"][0].shape == (1600,)

