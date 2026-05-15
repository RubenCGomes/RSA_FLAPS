from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from ml_app.mfls_app import MLApp
from ml_app.utils import record_microphone, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a short microphone sample and run MUSDB18HQ separation prediction.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Recording sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Number of microphone channels to capture")
    parser.add_argument("--device", default=None, help="Optional sounddevice input device index or name")
    parser.add_argument("--output-dir", default="runs/mic", help="Base directory for outputs")
    parser.add_argument("--checkpoint", default="runs/latest_train.pt", help="Optional model checkpoint to load")
    parser.add_argument("--data-root", default=None, help="Optional MUSDB18HQ dataset root")
    parser.add_argument("--base-filters", type=int, default=16, help="Must match the trained checkpoint architecture")
    parser.add_argument("--client-id", type=int, default=None, help="Optional client id for non-IID partitioning")
    parser.add_argument("--num-clients", type=int, default=None, help="Optional total number of clients")
    return parser.parse_args()


def load_checkpoint(app: MLApp, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        print(f"[warn] checkpoint not found: {checkpoint_path} -- using current model weights")
        return
    state_dict = torch.load(checkpoint_path, map_location=app.device, weights_only=True)
    app.model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    app = MLApp(
        data_root=args.data_root,
        base_filters=getattr(args, "base_filters", 16),
        client_id=args.client_id,
        num_clients=args.num_clients,
        sr=args.sample_rate,
    )
    app.set_run_path(str(run_dir))
    load_checkpoint(app, Path(args.checkpoint))

    print(f"Recording {args.seconds:.1f}s from the microphone at {args.sample_rate} Hz...")
    print("Tip: make a short sound after the countdown to test separation quality.")
    waveform = record_microphone(
        args.seconds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        device=args.device,
    )

    input_path = run_dir / "input.wav"
    save_wav(input_path, waveform, args.sample_rate)
    print(f"Saved input recording: {input_path}")

    outputs = app.predict(None, [waveform])
    print("Prediction outputs:")
    for item in outputs:
        print(item)

    print(f"Done. Separated stems are in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


