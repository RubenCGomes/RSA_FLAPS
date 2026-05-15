from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ml_app.mfls_app import MLApp


def _env_str(name: str, default: str | None = None) -> str | None:
	value = os.environ.get(name)
	if value is None:
		return default
	value = value.strip()
	return value if value else default


def _env_bool(name: str, default: bool | None = None) -> bool | None:
	value = os.environ.get(name)
	if value is None or value.strip() == "":
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
	value = os.environ.get(name)
	if value is None or value.strip() == "":
		return default
	try:
		return int(value)
	except ValueError:
		return default


def _env_float(name: str, default: float | None = None) -> float | None:
	value = os.environ.get(name)
	if value is None or value.strip() == "":
		return default
	try:
		return float(value)
	except ValueError:
		return default


def _resolve_data_root() -> str:
	for env_name in ("FLAPS_DATA_ROOT", "MUSDB_ROOT", "DATA_ROOT", "DATASET"):
		value = _env_str(env_name)
		if value:
			return value
	repo_root = Path(__file__).resolve().parents[3]
	return str(repo_root / "assets" / "datasets" / "musdb18hq")


def _resolve_device() -> str | None:
	device = _env_str("DEVICE")
	if device is None or device.lower() == "auto":
		return None
	return device


def _build_app() -> MLApp:
	full_tracks = _env_bool("FULL_TRACKS", default=False)
	segment_seconds = None if full_tracks else _env_float("SEGMENT_SECONDS", 5.0)

	return MLApp(
		data_root=_resolve_data_root(),
		split=_env_str("SPLIT", "train") or "train",
		sr=_env_int("SAMPLE_RATE", 16_000) or 16_000,
		segment_seconds=segment_seconds,
		device=_resolve_device(),
		n_fft=_env_int("N_FFT", 4_096) or 4_096,
		hop_length=_env_int("HOP_LENGTH", 1_024) or 1_024,
		base_filters=_env_int("BASE_FILTERS", 16) or 16,
		use_amp=_env_bool("USE_AMP", default=None),
		max_train_segments=_env_int("MAX_TRAIN_SEGMENTS", None),
		max_eval_segments=_env_int("MAX_EVAL_SEGMENTS", 32) or 32,
		client_id=_env_int("CLIENT_ID", None),
		num_clients=_env_int("NUM_CLIENTS", None),
		model_name=_env_str("MODEL_NAME", "unet_small") or "unet_small",
		augment_train=_env_bool("AUGMENT_TRAIN", default=False) or False,
		augment_gain_db=_env_float("AUGMENT_GAIN_DB", 6.0) or 6.0,
		augment_noise_std=_env_float("AUGMENT_NOISE_STD", 0.0) or 0.0,
		augment_stem_dropout_prob=_env_float("AUGMENT_STEM_DROPOUT_PROB", 0.0) or 0.0,
		augment_shift_seconds=_env_float("AUGMENT_SHIFT_SECONDS", 0.0) or 0.0,
		augment_remix_prob=_env_float("AUGMENT_REMIX_PROB", 0.0) or 0.0,
		augment_phase_flip=_env_bool("AUGMENT_PHASE_FLIP", default=False) or False,
	)


APP = _build_app()
model = APP.model
occupied = 0
current_run_path = ""


def in_usage(value):
	global occupied
	occupied = int(value)
	if occupied == 1:
		print("ML app is now in use, and blocked")
	elif occupied == 0:
		print("ML app is now freed")


def set_run_path(run_path):
	global current_run_path
	current_run_path = str(run_path)
	APP.set_run_path(current_run_path)
	print("Current run path set")


def get_parameters():
	return APP.get_parameters()


def set_parameters(parameters):
	global model
	APP.set_parameters(parameters)
	model = APP.model
	print("Parameters Set!")


def get_data():
	return APP.get_data()


def _merge_fit_config(config: dict | None) -> dict:
	merged = dict(config or {})
	merged.setdefault("train_chunk_duration", _env_float("TRAIN_CHUNK_DURATION", None))
	merged.setdefault("train_chunk_overlap", _env_float("TRAIN_CHUNK_OVERLAP", 0.0) or 0.0)
	merged.setdefault("loss_mask_weight", _env_float("LOSS_MASK_WEIGHT", 0.35) or 0.35)
	merged.setdefault("loss_mag_weight", _env_float("LOSS_MAG_WEIGHT", 0.35) or 0.35)
	merged.setdefault("loss_consistency_weight", _env_float("LOSS_CONSISTENCY_WEIGHT", 0.05) or 0.05)
	merged.setdefault("loss_kl_weight", _env_float("LOSS_KL_WEIGHT", 0.05) or 0.05)
	merged.setdefault("loss_sisdr_weight", _env_float("LOSS_SISDR_WEIGHT", 0.20) or 0.20)
	merged.setdefault("use_amp", _env_bool("USE_AMP", default=APP.use_amp) if APP.device.type == "cuda" else False)
	return merged


def fit(config):
	result = APP.fit(None, _merge_fit_config(config))
	if isinstance(result, tuple) and len(result) == 3:
		_, num_examples, metrics = result
		return num_examples, metrics
	if isinstance(result, tuple) and len(result) == 2:
		num_examples, metrics = result
		return num_examples, metrics
	raise TypeError(f"Unexpected fit result from MLApp: {type(result)!r}")


def _normalize_evaluate_result(result: Any) -> tuple[float, int, dict]:
	if isinstance(result, dict):
		loss = float(result.get("loss", 0.0))
		num_examples = int(result.get("n_examples", result.get("num_examples", 0)))
		return loss, num_examples, result
	if isinstance(result, tuple):
		if len(result) == 3:
			loss, num_examples, metrics = result
			return float(loss), int(num_examples), dict(metrics)
		if len(result) == 2:
			loss, metrics = result
			metrics_dict = dict(metrics) if isinstance(metrics, dict) else {"metrics": metrics}
			return float(loss), int(metrics_dict.get("n_examples", 0)), metrics_dict
	raise TypeError(f"Unexpected evaluate result from MLApp: {type(result)!r}")


def evaluate(config):
	merged = dict(config or {})
	merged.setdefault("split", _env_str("EVAL_SPLIT", "val") or "val")
	merged.setdefault("batch_size", _env_int("EVAL_BATCH_SIZE", 1) or 1)
	merged.setdefault("max_segments", _env_int("MAX_EVAL_SEGMENTS", APP.max_eval_segments))
	merged.setdefault("use_amp", _env_bool("USE_AMP", default=APP.use_amp) if APP.device.type == "cuda" else False)
	merged.setdefault("loss_mask_weight", _env_float("LOSS_MASK_WEIGHT", 0.35) or 0.35)
	merged.setdefault("loss_mag_weight", _env_float("LOSS_MAG_WEIGHT", 0.35) or 0.35)
	merged.setdefault("loss_consistency_weight", _env_float("LOSS_CONSISTENCY_WEIGHT", 0.05) or 0.05)
	merged.setdefault("loss_kl_weight", _env_float("LOSS_KL_WEIGHT", 0.05) or 0.05)
	merged.setdefault("loss_sisdr_weight", _env_float("LOSS_SISDR_WEIGHT", 0.20) or 0.20)
	result = APP.evaluate(None, merged)
	return _normalize_evaluate_result(result)


def _preview_input_path() -> str | None:
	data = APP.get_data({"max_segments": 1, "max_eval_segments": 1})
	preview = data.get("preview") if isinstance(data, dict) else None
	if not preview:
		return None
	first = preview[0]
	files = first.get("files", {}) if isinstance(first, dict) else {}
	return files.get("mixture")


def predict(plot_graphs, before_after, centralized_predict="false"):
	input_path = _preview_input_path()
	outputs = []
	if input_path:
		predict_config: dict[str, Any] = {}
		chunk_duration = _env_float("PREDICT_CHUNK_DURATION", 30.0)
		if chunk_duration is not None:
			predict_config["chunk_duration"] = chunk_duration
		predict_config["use_amp"] = _env_bool("USE_AMP", default=APP.use_amp) if APP.device.type == "cuda" else False
		outputs = APP.predict(None, [input_path], config=predict_config)

	losses = {"loss": 0.0, "mode": str(before_after)}
	metrics = {
		"centralized_predict": str(centralized_predict),
		"plot_graphs": str(plot_graphs),
		"before_after": str(before_after),
		"outputs": outputs,
	}
	return losses, metrics


if __name__ == "__main__":
	print(get_data())

