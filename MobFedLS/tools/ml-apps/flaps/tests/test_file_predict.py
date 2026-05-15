from __future__ import annotations

import types
from pathlib import Path

import numpy as np


def test_predict_from_file_main_uses_file_path_and_predict(monkeypatch, tmp_path):
    import scripts.predict_from_file as file_script

    captured = {}

    class DummyApp:
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.model = types.SimpleNamespace(load_state_dict=lambda state: None)
            self.run_path = None

        def set_run_path(self, run_path):
            self.run_path = Path(run_path)

        def predict(self, parameters, inputs, config=None):
            captured["parameters"] = parameters
            captured["inputs"] = inputs
            return [{"input": "file", "stems": {"vocals": "vocals.wav"}}]

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(file_script, "MLApp", DummyApp)
    monkeypatch.setattr(file_script, "load_checkpoint", lambda app, checkpoint_path: None)
    monkeypatch.setattr(file_script, "parse_args", lambda: types.SimpleNamespace(
        audio_path=str(audio_path),
        seconds=None,
        output_dir=str(tmp_path / "file-runs"),
        checkpoint=str(tmp_path / "missing.pt"),
        data_root=None,
        sample_rate=16000,
        model="unet_small",
        base_filters=16,
        client_id=None,
        num_clients=None,
        device="gpu",
        chunk_duration=30.0,
        enable_amp=False,
    ))

    exit_code = file_script.main()

    assert exit_code == 0
    assert captured["parameters"] is None
    assert isinstance(captured["inputs"], list)
    assert captured["inputs"][0] == str(audio_path)

