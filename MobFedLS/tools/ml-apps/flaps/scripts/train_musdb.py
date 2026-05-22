from __future__ import annotations

import argparse
import configparser
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CMD_DIR = Path(__file__).resolve().parents[1] / "cmd"
if str(CMD_DIR) not in sys.path:
    sys.path.insert(0, str(CMD_DIR))

import torch

from ml_app.mfls_app import MLApp


DEFAULT_CONFIG: dict[str, Any] = {
    "data_root": "musdb18hq/",
    "output_dir": "runs/train",
    "epochs": 10,
    "batch_size": 1,
    "accumulation_steps": 4,
    "learning_rate": 1e-3,
    "segment_seconds": 5.0,
    "full_tracks": False,
    "train_chunk_duration": None,
    "train_chunk_overlap": 0.0,
    "sample_rate": 16_000,
    "model": "unet_small",
    "base_filters": 16,
    "max_train_segments": None,
    "max_eval_segments": 16,
    "client_id": None,
    "num_clients": None,
    "resume": None,
    "patience": 5,
    "use_amp": True,
    "augment_train": True,
    "augment_gain_db": 6.0,
    "augment_noise_std": 0.001,
    "augment_stem_dropout_prob": 0.05,
    "augment_shift_seconds": 0.2,
    "augment_remix_prob": 0.1,
    "augment_phase_flip": True,
    "loss_mask_weight": 0.35,
    "loss_mag_weight": 0.35,
    "loss_consistency_weight": 0.05,
    "loss_kl_weight": 0.05,
    "loss_sisdr_weight": 0.20,
    "validation_mode": "quick",
    "validation_split": "val",
    "strict_split_check": True,
    "scheduler": "cosine",
    "scheduler_step_size": 3,
    "scheduler_gamma": 0.5,
    "scheduler_t_max": 10,
    "scheduler_min_lr": 1e-6,
    "scheduler_mode": "max",
}


def _load_config_file(path_str: str) -> dict[str, Any]:
    config_path = Path(path_str)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() != ".ini":
        raise ValueError(f"Config file must be .ini, got '{config_path.suffix or '<no suffix>'}'.")

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(config_path, encoding="utf-8")
    except configparser.Error as exc:
        raise ValueError(f"Invalid INI in config file '{config_path}': {exc}") from exc

    allowed_sections = {"train"}
    unknown_sections = sorted(section for section in parser.sections() if section not in allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown sections in '{config_path}': {unknown_sections}. Allowed: ['train']")

    raw_values: dict[str, str] = dict(parser.defaults())
    if parser.has_section("train"):
        raw_values.update(dict(parser.items("train")))

    config_data = {key: _coerce_config_value(key, value) for key, value in raw_values.items()}

    unknown_keys = sorted(set(config_data.keys()) - set(DEFAULT_CONFIG.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in '{config_path}': {unknown_keys}")
    return config_data


def _coerce_config_value(key: str, raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()
    null_tokens = {"", "none", "null"}

    bool_keys = {
        "full_tracks",
        "use_amp",
        "augment_train",
        "augment_phase_flip",
        "strict_split_check",
    }
    int_keys = {
        "epochs",
        "batch_size",
        "accumulation_steps",
        "sample_rate",
        "base_filters",
        "max_eval_segments",
        "patience",
        "scheduler_step_size",
        "scheduler_t_max",
    }
    float_keys = {
        "learning_rate",
        "segment_seconds",
        "train_chunk_overlap",
        "augment_gain_db",
        "augment_noise_std",
        "augment_stem_dropout_prob",
        "augment_shift_seconds",
        "augment_remix_prob",
        "loss_mask_weight",
        "loss_mag_weight",
        "loss_consistency_weight",
        "loss_kl_weight",
        "loss_sisdr_weight",
        "scheduler_gamma",
        "scheduler_min_lr",
    }
    nullable_int_keys = {"max_train_segments", "client_id", "num_clients"}
    nullable_float_keys = {"train_chunk_duration"}
    nullable_str_keys = {"resume"}

    if key in bool_keys:
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean for '{key}': '{raw_value}'")
    if key in int_keys:
        return int(value)
    if key in float_keys:
        return float(value)
    if key in nullable_int_keys:
        return None if lowered in null_tokens else int(value)
    if key in nullable_float_keys:
        return None if lowered in null_tokens else float(value)
    if key in nullable_str_keys:
        return None if lowered in null_tokens else value
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger MUSDB18HQ source separation model.")
    parser.add_argument("--config", type=str, default=None, help="Optional INI config file path")
    parser.add_argument("--data-root", default=argparse.SUPPRESS, help="Path to musdb18hq root")
    parser.add_argument("--output-dir", default=argparse.SUPPRESS, help="Directory for training artifacts")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Training batch size")
    parser.add_argument("--accumulation-steps", type=int, default=argparse.SUPPRESS, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS, help="Learning rate")
    parser.add_argument("--segment-seconds", type=float, default=argparse.SUPPRESS, help="Segment length in seconds")
    parser.add_argument("--full-tracks", action="store_true", default=argparse.SUPPRESS, help="Use full-song tracks instead of fixed-length segments")
    parser.add_argument("--train-chunk-duration", type=float, default=argparse.SUPPRESS, help="Optional training chunk duration in seconds for memory-efficient training")
    parser.add_argument("--train-chunk-overlap", type=float, default=argparse.SUPPRESS, help="Training chunk overlap ratio in [0, 1) when chunking is enabled")
    parser.add_argument("--sample-rate", type=int, default=argparse.SUPPRESS, help="Target sample rate")
    parser.add_argument("--model", type=str, default=argparse.SUPPRESS, choices=["unet_small", "unet_large"], help="Model architecture")
    parser.add_argument("--base-filters", type=int, default=argparse.SUPPRESS, help="Base channel width for the U-Net")
    parser.add_argument("--max-train-segments", type=int, default=argparse.SUPPRESS, help="Optional training subset size")
    parser.add_argument("--max-eval-segments", type=int, default=argparse.SUPPRESS, help="Validation subset size")
    parser.add_argument("--client-id", type=int, default=argparse.SUPPRESS, help="Optional client id for non-IID partitioning")
    parser.add_argument("--num-clients", type=int, default=argparse.SUPPRESS, help="Optional total number of clients")
    parser.add_argument("--resume", default=argparse.SUPPRESS, help="Optional checkpoint to resume from")
    parser.add_argument("--patience", type=int, default=argparse.SUPPRESS, help="Early stopping patience on SI-SDR improvement")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Use mixed precision on CUDA")
    parser.add_argument("--augment-train", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Enable train-time waveform augmentation")
    parser.add_argument("--augment-gain-db", type=float, default=argparse.SUPPRESS, help="Random stem gain range in dB (±value)")
    parser.add_argument("--augment-noise-std", type=float, default=argparse.SUPPRESS, help="Gaussian noise std added to augmented mixtures")
    parser.add_argument("--augment-stem-dropout-prob", type=float, default=argparse.SUPPRESS, help="Probability of muting one stem during augmentation")
    parser.add_argument("--augment-shift-seconds", type=float, default=argparse.SUPPRESS, help="Max absolute random stem shift in seconds")
    parser.add_argument("--augment-remix-prob", type=float, default=argparse.SUPPRESS, help="Probability of shuffling stems between tracks")
    parser.add_argument("--augment-phase-flip", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Randomly flip signal phase")
    parser.add_argument("--loss-mask-weight", type=float, default=argparse.SUPPRESS, help="Weight for mask regression loss")
    parser.add_argument("--loss-mag-weight", type=float, default=argparse.SUPPRESS, help="Weight for multi-scale magnitude loss")
    parser.add_argument("--loss-consistency-weight", type=float, default=argparse.SUPPRESS, help="Weight for mask sum-to-one consistency loss")
    parser.add_argument("--loss-kl-weight", type=float, default=argparse.SUPPRESS, help="Weight for KL loss on mask distributions")
    parser.add_argument("--loss-sisdr-weight", type=float, default=argparse.SUPPRESS, help="Weight for spectral SI-SDR loss")
    parser.add_argument("--validation-mode", type=str, default=argparse.SUPPRESS, choices=["quick", "full"], help="Validation scope: subset (quick) or full split")
    parser.add_argument("--validation-split", type=str, default=argparse.SUPPRESS, choices=["val", "test"], help="Split used for early stopping/selection")
    parser.add_argument("--strict-split-check", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Fail if train/validation track overlap is detected")
    parser.add_argument("--scheduler", type=str, default=argparse.SUPPRESS, choices=["none", "step", "cosine", "plateau"], help="Learning-rate scheduler")
    parser.add_argument("--scheduler-step-size", type=int, default=argparse.SUPPRESS, help="Step interval / patience (scheduler dependent)")
    parser.add_argument("--scheduler-gamma", type=float, default=argparse.SUPPRESS, help="LR decay factor")
    parser.add_argument("--scheduler-t-max", type=int, default=argparse.SUPPRESS, help="Cosine scheduler cycle length")
    parser.add_argument("--scheduler-min-lr", type=float, default=argparse.SUPPRESS, help="Minimum learning rate")
    parser.add_argument("--scheduler-mode", type=str, default=argparse.SUPPRESS, choices=["max", "min"], help="Plateau scheduler metric mode")
    return parser.parse_args()


def resolve_args(parsed: argparse.Namespace) -> argparse.Namespace:
    merged = dict(DEFAULT_CONFIG)
    config_path = getattr(parsed, "config", None)
    if config_path:
        merged.update(_load_config_file(str(config_path)))

    cli_values = vars(parsed).copy()
    cli_values.pop("config", None)
    merged.update(cli_values)
    return argparse.Namespace(**merged)


def load_checkpoint(app: MLApp, checkpoint: Path) -> None:
    state_dict = torch.load(checkpoint, map_location=app.device, weights_only=True)
    app.model.load_state_dict(state_dict)


def main() -> int:
    args = resolve_args(parse_args())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    app = MLApp(
        data_root=args.data_root,
        sr=args.sample_rate,
        segment_seconds=None if args.full_tracks else args.segment_seconds,
        base_filters=args.base_filters,
        model_name=args.model,
        use_amp=args.use_amp,
        max_train_segments=args.max_train_segments,
        max_eval_segments=args.max_eval_segments,
        client_id=args.client_id,
        num_clients=args.num_clients,
        augment_train=args.augment_train,
        augment_gain_db=args.augment_gain_db,
        augment_noise_std=args.augment_noise_std,
        augment_stem_dropout_prob=args.augment_stem_dropout_prob,
        augment_shift_seconds=args.augment_shift_seconds,
        augment_remix_prob=args.augment_remix_prob,
        augment_phase_flip=args.augment_phase_flip,
    )
    app.set_run_path(str(run_dir))

    train_tracks = app.split_track_names("train", max_segments=args.max_train_segments)
    validation_track_limit = None if args.validation_mode == "full" else args.max_eval_segments
    val_tracks = app.split_track_names(args.validation_split, max_segments=validation_track_limit)
    overlap = sorted(train_tracks.intersection(val_tracks))
    if overlap:
        message = f"Detected train/{args.validation_split} overlap on {len(overlap)} tracks"
        if args.strict_split_check:
            raise RuntimeError(f"{message}: {overlap[:5]}")
        print(f"[warn] {message}: {overlap[:5]}")

    if args.resume:
        checkpoint = Path(args.resume)
        if checkpoint.exists():
            load_checkpoint(app, checkpoint)
            print(f"Resumed from: {checkpoint}")
        else:
            print(f"[warn] resume checkpoint not found: {checkpoint}")

    config = {
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "use_amp": args.use_amp,
        "train_chunk_duration": args.train_chunk_duration,
        "train_chunk_overlap": args.train_chunk_overlap,
        "loss_mask_weight": args.loss_mask_weight,
        "loss_mag_weight": args.loss_mag_weight,
        "loss_consistency_weight": args.loss_consistency_weight,
        "loss_kl_weight": args.loss_kl_weight,
        "loss_sisdr_weight": args.loss_sisdr_weight,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "scheduler_t_max": args.scheduler_t_max,
        "scheduler_min_lr": args.scheduler_min_lr,
        "scheduler_mode": args.scheduler_mode,
    }
    if args.max_train_segments is not None:
        config["max_segments"] = args.max_train_segments

    metrics_log = []
    best_score = float("-inf")
    best_epoch = -1
    best_path = run_dir / "best.pt"
    patience_left = args.patience

    print(f"Training run directory: {run_dir}")
    print(f"Using device: {app.device}")
    print(f"Model: {args.model}")
    print("Starting training...")

    eval_max_segments = None if args.validation_mode == "full" else args.max_eval_segments
    for epoch in range(1, args.epochs + 1):
        trained_params, n_examples, train_metrics = app.fit(app.get_parameters(), {**config, "epochs": 1})
        val_metrics = app.evaluate(
            trained_params,
            {
                "split": args.validation_split,
                "batch_size": 2,
                "max_segments": eval_max_segments,
                "use_amp": args.use_amp,
                "loss_mask_weight": args.loss_mask_weight,
                "loss_mag_weight": args.loss_mag_weight,
                "loss_consistency_weight": args.loss_consistency_weight,
                "loss_kl_weight": args.loss_kl_weight,
                "loss_sisdr_weight": args.loss_sisdr_weight,
            },
        )
        score = float(val_metrics.get("mean_si_sdr_improvement", float("-inf")))

        if args.scheduler == "plateau":
            scheduler_metric = score if args.scheduler_mode == "max" else float(val_metrics.get("loss", 0.0))
            app.update_scheduler(scheduler_metric)

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "n_examples": n_examples,
            "score": score,
            "learning_rate": float(train_metrics.get("learning_rate", args.learning_rate)),
        }
        metrics_log.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        app.save_checkpoint("latest.pt")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            app.save_checkpoint("best.pt")
            patience_left = args.patience
            print(f"[best] epoch={epoch} score={best_score:.4f}")
        else:
            patience_left -= 1
            print(f"[patience] remaining={patience_left}")
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_ran": len(metrics_log),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

