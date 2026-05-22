from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

CMD_DIR = Path(__file__).resolve().parents[1] / "cmd"
if str(CMD_DIR) not in sys.path:
    sys.path.insert(0, str(CMD_DIR))

import torch

from ml_app.mfls_app import MLApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MUSDB18HQ source separation on an audio file.")
    parser.add_argument("--audio-path", required=True, help="Path to the input audio file")
    parser.add_argument("--seconds", type=float, default=None, help="Optional duration hint for future extensions")
    parser.add_argument("--output-dir", default="runs/file", help="Base directory for outputs")
    parser.add_argument("--checkpoint", default="runs/latest_train.pt", help="Optional model checkpoint to load")
    parser.add_argument("--data-root", default=None, help="Optional MUSDB18HQ dataset root")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for the model")
    parser.add_argument("--model", type=str, default="unet_small", choices=["unet_small", "unet_large"], help="Model architecture to use")
    parser.add_argument("--base-filters", type=int, default=16, help="Must match the trained checkpoint architecture")
    parser.add_argument("--client-id", type=int, default=None, help="Optional client id for non-IID partitioning")
    parser.add_argument("--num-clients", type=int, default=None, help="Optional total number of clients")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--chunk-duration", type=float, default=30.0, help="Process audio in chunks of this duration (seconds) to save memory")
    parser.add_argument("--enable-amp", action="store_true", help="Enable automatic mixed precision (AMP) for GPU inference")
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
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # Determine device
    requested_device = args.device if args.device in {"auto", "cpu", "cuda"} else "auto"
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Apply PyTorch CUDA memory optimization if using GPU
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    app = MLApp(
        data_root=args.data_root,
        base_filters=getattr(args, "base_filters", 16),
        client_id=args.client_id,
        num_clients=args.num_clients,
        sr=args.sample_rate,
        device=device,
        use_amp=args.enable_amp if device == "cuda" else False,
        model_name=args.model,
    )
    app.set_run_path(str(run_dir))
    load_checkpoint(app, Path(args.checkpoint))

    print(f"Input audio: {audio_path}")
    print(f"Output directory: {run_dir}")

    # Use chunked prediction for better memory efficiency
    predict_config = {"chunk_duration": args.chunk_duration} if hasattr(args, "chunk_duration") else {}
    outputs = app.predict(None, [str(audio_path)], config=predict_config)

    print("Prediction outputs:")
    for item in outputs:
        print(item)

    print(f"Done. Separated stems are in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

