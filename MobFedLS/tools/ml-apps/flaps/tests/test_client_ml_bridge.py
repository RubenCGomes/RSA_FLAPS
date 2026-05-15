from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clientML


class DummyApp:
    def __init__(self):
        self.device = SimpleNamespace(type="cpu")
        self.use_amp = False
        self.max_eval_segments = 8
        self.model = SimpleNamespace(name="dummy-model")
        self.calls: list[tuple] = []
        self.run_path = None

    def get_parameters(self):
        self.calls.append(("get_parameters",))
        return [np.array([1.0], dtype=np.float32)]

    def set_parameters(self, parameters):
        self.calls.append(("set_parameters", parameters))

    def get_data(self, selection=None):
        self.calls.append(("get_data", selection))
        return {"preview": [{"files": {"mixture": "/tmp/mix.wav"}}]}

    def fit(self, parameters, config=None):
        self.calls.append(("fit", parameters, config))
        return [np.array([2.0], dtype=np.float32)], 7, {"loss": 1.5}

    def evaluate(self, parameters, config=None):
        self.calls.append(("evaluate", parameters, config))
        return {"loss": 0.25, "n_examples": 4, "mean_si_sdr": 9.0}

    def predict(self, parameters, inputs, config=None):
        self.calls.append(("predict", parameters, inputs, config))
        return [{"input": inputs[0], "output_dir": "/tmp/out", "stems": {"vocals": "vocals.wav"}}]

    def set_run_path(self, run_path):
        self.calls.append(("set_run_path", run_path))
        self.run_path = Path(run_path)


def test_client_ml_bridge_roundtrip(monkeypatch, tmp_path):
    dummy = DummyApp()
    monkeypatch.setattr(clientML, "APP", dummy)
    monkeypatch.setattr(clientML, "model", dummy.model)

    clientML.set_run_path(tmp_path / "run0")
    params = clientML.get_parameters()
    clientML.set_parameters(params)
    data = clientML.get_data()
    num_examples, metrics = clientML.fit({"epochs": 1, "batch_size": 1})
    loss, eval_examples, eval_metrics = clientML.evaluate({"split": "val"})
    losses, predict_metrics = clientML.predict("false", "before")

    assert dummy.run_path == tmp_path / "run0"
    assert isinstance(data, dict)
    assert num_examples == 7
    assert np.isclose(metrics["loss"], 1.5)
    assert np.isclose(loss, 0.25)
    assert eval_examples == 4
    assert np.isclose(eval_metrics["mean_si_sdr"], 9.0)
    assert np.isclose(losses["loss"], 0.0)
    assert predict_metrics["outputs"]
    assert dummy.calls[0][0] == "set_run_path"
    assert any(call[0] == "predict" for call in dummy.calls)



