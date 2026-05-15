from __future__ import annotations

from pathlib import Path

from ml_app.mfls_app import MLApp


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_mfls_app_end_to_end_smoke(tmp_path):
    dataset_root = repo_root() / "assets" / "datasets" / "musdb18hq"
    assert dataset_root.exists(), "musdb18hq dataset is required for this smoke test"

    app = MLApp(data_root=str(dataset_root), max_train_segments=2, max_eval_segments=1, device="cpu", use_amp=False)
    app.set_run_path(str(tmp_path / "runs"))

    data_info = app.get_data({"max_segments": 2, "max_eval_segments": 1})
    assert data_info["train"]["segments"] == 2

    params = app.get_parameters()
    trained_params, n_examples, metrics = app.fit(params, {"epochs": 1, "batch_size": 1, "max_segments": 2})
    assert n_examples == 2
    assert "loss" in metrics

    eval_metrics = app.evaluate(trained_params, {"split": "val", "max_segments": 1, "batch_size": 1})
    assert eval_metrics["n_examples"] == 1
    assert "mean_si_sdr" in eval_metrics

    preview = data_info["preview"][0]["files"]["mixture"]
    preds = app.predict(trained_params, [preview])
    assert len(preds) == 1
    assert set(preds[0]["stems"].keys()) == {"vocals", "drums", "bass", "other"}

