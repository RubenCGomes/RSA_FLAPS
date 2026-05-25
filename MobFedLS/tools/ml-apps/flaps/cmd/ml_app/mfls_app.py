from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from .dataset.musdb_loader import MusdbSeparationDataset
from .models.unet_large import UNetLarge
from .models.unet_small import UNetSmall
from .utils.audio import istft_from_mag_phase, si_sdr, stem_names
from .utils.losses import composite_separation_loss, stem_classification_loss


class MLApp:
    """MFLS-compatible ML-App for MUSDB18HQ source separation."""

    def __init__(
        self,
        data_root: str | None = None,
        split: str = "train",
        sr: int = 44_100,
        segment_seconds: float | None = 5.0,
        device: str | None = None,
        n_fft: int = 4096,
        hop_length: int = 1024,
        base_filters: int = 16,
        use_amp: bool | None = None,
        max_train_segments: int | None = None,
        max_eval_segments: int = 32,
        client_id: int | None = None,
        num_clients: int | None = None,
        model_name: str = "unet_small",
        augment_train: bool = False,
        augment_gain_db: float = 6.0,
        augment_noise_std: float = 0.0,
        augment_stem_dropout_prob: float = 0.0,
        augment_shift_seconds: float = 0.0,
        augment_remix_prob: float = 0.0,
        augment_phase_flip: bool = False,
        target_stem: str | None = None,
    ):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_root = Path(data_root or os.environ.get("MUSDB_ROOT", self.repo_root / "musdb18hq"))
        self.split = split
        self.sr = int(sr)
        # allow None to indicate using whole tracks
        self.segment_seconds = None if segment_seconds is None else float(segment_seconds)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.base_filters = int(base_filters)
        self.max_train_segments = max_train_segments
        self.max_eval_segments = max_eval_segments
        self.client_id = client_id if client_id is not None else _env_int("CLIENT_ID")
        self.num_clients = num_clients if num_clients is not None else _env_int("NUM_CLIENTS")
        self.model_name = model_name
        self.augment_train = bool(augment_train)
        self.augment_gain_db = float(augment_gain_db)
        self.augment_noise_std = float(augment_noise_std)
        self.augment_stem_dropout_prob = float(augment_stem_dropout_prob)
        self.augment_shift_seconds = float(augment_shift_seconds)
        self.augment_remix_prob = float(augment_remix_prob)
        self.augment_phase_flip = bool(augment_phase_flip)
        raw_target = (target_stem or os.environ.get("TARGET_STEM", "")).strip() or None
        if raw_target is not None and raw_target not in stem_names():
            raise ValueError(f"TARGET_STEM {raw_target!r} must be one of {stem_names()}")
        self.target_stem: str | None = raw_target
        self.target_stem_index: int | None = stem_names().index(raw_target) if raw_target else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = _env_bool("USE_AMP", default=self.device.type == "cuda") if use_amp is None else bool(use_amp)
        self.model = self._build_model(model_name)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler: StepLR | CosineAnnealingLR | ReduceLROnPlateau | None = None
        self._scheduler_signature: tuple | None = None
        self.run_path = Path(os.environ.get("RUN_PATH", self.repo_root / "runs"))
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._cache: Dict[str, MusdbSeparationDataset] = {}
        self._training_history: list[dict] = []
        self._round: int = 0

    def _build_model(self, model_name: str):
        """Build model based on model_name."""
        if model_name == "unet_small":
            return UNetSmall(in_channels=1, out_channels=4, base_filters=self.base_filters).to(self.device)
        if model_name == "unet_large":
            return UNetLarge(in_channels=1, out_channels=4, base_filters=max(24, self.base_filters)).to(self.device)
        raise ValueError(f"Unknown model: {model_name}. Available models: unet_small, unet_large")

    def get_parameters(self) -> List[np.ndarray]:
        return [tensor.detach().cpu().numpy() for tensor in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        if len(parameters) != len(keys):
            raise ValueError(f"Parameter count mismatch: expected {len(keys)}, received {len(parameters)}")
        updated = {}
        for key, array in zip(keys, parameters):
            np_array = np.asarray(array)
            # Some decoded payloads are read-only views; copy to guarantee safe tensor writes.
            if not np_array.flags.writeable:
                np_array = np_array.copy()
            tensor = torch.as_tensor(np_array, dtype=state_dict[key].dtype)
            updated[key] = tensor
        self.model.load_state_dict(updated)

    def load_checkpoint_from_bytes(self, data: bytes) -> None:
        import io
        buf = io.BytesIO(data)
        state_dict = torch.load(buf, map_location=self.device, weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)

    def get_data(self, selection: dict | None = None) -> dict:
        selection = selection or {}
        train_dataset = self._dataset("train", max_segments=selection.get("max_segments", self.max_train_segments))
        val_dataset = self._dataset("val", max_segments=selection.get("max_eval_segments", self.max_eval_segments))
        test_dataset = self._dataset("test", max_segments=selection.get("max_eval_segments", self.max_eval_segments))
        return {
            "train": train_dataset.summary(),
            "val": val_dataset.summary(),
            "test": test_dataset.summary(),
            "preview": train_dataset.preview(limit=3),
        }

    def fit(self, parameters: List[np.ndarray] | None, config: dict | None = None):
        with self.lock:
            return self._fit(parameters, config)

    def _fit(self, parameters: List[np.ndarray] | None, config: dict | None = None):
        _fit_start = time.monotonic()
        self._round += 1
        _current_round = self._round
        config = {} if config is None else dict(config)
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", 1))
        learning_rate = float(config.get("learning_rate", 1e-3))
        accumulation_steps = max(1, int(config.get("accumulation_steps", 1)))
        max_grad_norm = float(config.get("max_grad_norm", 1.0))
        use_amp = bool(config.get("use_amp", self.use_amp)) and self.device.type == "cuda"
        loss_mask_weight = float(config.get("loss_mask_weight", 0.35))
        loss_mag_weight = float(config.get("loss_mag_weight", 0.35))
        loss_consistency_weight = float(config.get("loss_consistency_weight", 0.05))
        loss_kl_weight = float(config.get("loss_kl_weight", 0.05))
        loss_sisdr_weight = float(config.get("loss_sisdr_weight", 0.20))
        loss_cls_weight = float(config.get("loss_cls_weight", 0.1))
        train_chunk_duration = config.get("train_chunk_duration")
        train_chunk_overlap = float(config.get("train_chunk_overlap", 0.0))
        if not 0.0 <= train_chunk_overlap < 1.0:
            raise ValueError("train_chunk_overlap must be in [0.0, 1.0).")
        train_chunk_frames = self._chunk_frames_from_duration(train_chunk_duration)
        max_segments = config.get("max_segments", self.max_train_segments)
        if parameters is not None and len(parameters) > 0:
            self.set_parameters(parameters)

        self._configure_scheduler(config, learning_rate)
        dataset = self._dataset("train", max_segments=max_segments)
        if len(dataset) == 0:
            return self.get_parameters(), 0, {
                "loss": 0.0,
                "mask_loss": 0.0,
                "mag_loss": 0.0,
                "consistency_loss": 0.0,
                "kl_loss": 0.0,
            }

        loader = self._loader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_mag_loss = 0.0
        total_consistency = 0.0
        total_kl = 0.0
        total_sisdr_loss = 0.0
        total_cls_loss = 0.0
        total_cls_correct = 0.0
        steps = 0
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.optimizer.zero_grad(set_to_none=True)

        for _epoch in range(epochs):
            for batch_index, batch in enumerate(loader, start=1):
                mix_mag = batch["mix_mag"].to(self.device)
                mix_mag_raw = batch["mix_mag_raw"].to(self.device)
                target_masks = batch["target_masks"].to(self.device)
                target_mags = batch["target_mags"].to(self.device)

                # Presence labels: 1 if stem energy > threshold, 0 if dropped by augmentation
                stem_energy = target_mags.mean(dim=(-2, -1))  # (batch, 4)
                presence_labels = (stem_energy > 1e-4).float()

                chunk_ranges = self._build_chunk_ranges(
                    total_frames=int(mix_mag.shape[-1]),
                    chunk_frames=train_chunk_frames,
                    overlap_ratio=train_chunk_overlap,
                )
                num_chunks = max(1, len(chunk_ranges))
                batch_loss = 0.0
                batch_mask_loss = 0.0
                batch_mag_loss = 0.0
                batch_consistency_loss = 0.0
                batch_kl_loss = 0.0
                batch_sisdr_loss = 0.0

                for start_frame, end_frame in chunk_ranges:
                    mix_mag_chunk = mix_mag[..., start_frame:end_frame]
                    mix_mag_raw_chunk = mix_mag_raw[..., start_frame:end_frame]
                    target_masks_chunk = target_masks[..., start_frame:end_frame]
                    target_mags_chunk = target_mags[..., start_frame:end_frame]

                    device_type = self.device.type if self.device.type == "cpu" else "cuda"
                    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
                    with autocast_ctx:
                        pred_masks, cls_logits = self.model(mix_mag_chunk)
                        est_mags = pred_masks * mix_mag_raw_chunk.unsqueeze(1)
                        if self.target_stem_index is not None:
                            idx = self.target_stem_index
                            pred_masks_sup = pred_masks[:, idx:idx+1]
                            target_masks_sup = target_masks_chunk[:, idx:idx+1]
                            est_mags_sup = est_mags[:, idx:idx+1]
                            target_mags_sup = target_mags_chunk[:, idx:idx+1]
                            consistency_w = 0.0
                        else:
                            pred_masks_sup = pred_masks
                            target_masks_sup = target_masks_chunk
                            est_mags_sup = est_mags
                            target_mags_sup = target_mags_chunk
                            consistency_w = loss_consistency_weight
                        loss, components = composite_separation_loss(
                            pred_masks_sup,
                            target_masks_sup,
                            est_mags_sup,
                            target_mags_sup,
                            mask_weight=loss_mask_weight,
                            mag_weight=loss_mag_weight,
                            consistency_weight=consistency_w,
                            kl_weight=loss_kl_weight,
                            sisdr_weight=loss_sisdr_weight,
                        )
                        mask_loss = components["mask_loss"]
                        mag_loss = components["mag_loss"]
                        consistency_loss = components["consistency_loss"]
                        kl_loss = components["kl_loss"]
                        sisdr_l = components["sisdr_loss"]
                        if self.target_stem_index is not None and loss_cls_weight > 0:
                            cls_loss = stem_classification_loss(cls_logits, presence_labels, self.target_stem_index)
                            loss = loss + loss_cls_weight * cls_loss
                            with torch.no_grad():
                                cls_pred = torch.sigmoid(cls_logits[:, self.target_stem_index]) > 0.5
                                cls_correct = (cls_pred == presence_labels[:, self.target_stem_index].bool()).float().mean()
                        else:
                            cls_loss = torch.zeros(1, device=self.device)
                            cls_correct = torch.zeros(1, device=self.device)

                    # Average chunk contributions so one batch has comparable gradient scale
                    scaled_loss = loss / (accumulation_steps * num_chunks)
                    if use_amp:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    batch_loss += float(loss.detach().item())
                    batch_mask_loss += float(mask_loss.detach().item())
                    batch_mag_loss += float(mag_loss.detach().item())
                    batch_consistency_loss += float(consistency_loss.detach().item())
                    batch_kl_loss += float(kl_loss.detach().item())
                    batch_sisdr_loss += float(sisdr_l.detach().item())
                    total_cls_loss += float(cls_loss.detach().item()) / num_chunks
                    total_cls_correct += float(cls_correct.detach().item()) / num_chunks

                if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += batch_loss / num_chunks
                total_mask_loss += batch_mask_loss / num_chunks
                total_mag_loss += batch_mag_loss / num_chunks
                total_consistency += batch_consistency_loss / num_chunks
                total_kl += batch_kl_loss / num_chunks
                total_sisdr_loss += batch_sisdr_loss / num_chunks
                steps += 1

            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        metrics = {
            "loss": total_loss / max(1, steps),
            "mask_loss": total_mask_loss / max(1, steps),
            "mag_loss": total_mag_loss / max(1, steps),
            "consistency_loss": total_consistency / max(1, steps),
            "kl_loss": total_kl / max(1, steps),
            "sisdr_loss": total_sisdr_loss / max(1, steps),
            "cls_loss": total_cls_loss / max(1, steps),
            "cls_accuracy": total_cls_correct / max(1, steps),
            "epochs": epochs,
            "batch_size": batch_size,
            "n_examples": len(dataset),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "target_stem": self.target_stem or "all",
        }
        self._training_history.append({
            "round": _current_round,
            "phase": "fit",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _fit_start, 1),
            "loss": metrics["loss"],
            "cls_loss": metrics["cls_loss"],
            "cls_accuracy": metrics["cls_accuracy"],
            "n_examples": metrics["n_examples"],
        })
        self._save_checkpoint("latest_train.pt")
        return self.get_parameters(), len(dataset), metrics

    def evaluate(self, parameters: List[np.ndarray] | None, *args, **kwargs) -> dict:
        with self.lock:
            return self._evaluate(parameters, *args, **kwargs)

    def _evaluate(self, parameters: List[np.ndarray] | None, *args, **kwargs) -> dict:
        _eval_start = time.monotonic()
        _current_round = self._round
        config_arg = args[0] if args else kwargs.get("config")
        config = {} if config_arg is None else dict(config_arg)
        split = config.get("split", "val")
        batch_size = int(config.get("batch_size", 1))
        max_segments = config.get("max_segments", self.max_eval_segments)
        use_amp = bool(config.get("use_amp", self.use_amp)) and self.device.type == "cuda"
        loss_mask_weight = float(config.get("loss_mask_weight", 0.35))
        loss_mag_weight = float(config.get("loss_mag_weight", 0.35))
        loss_consistency_weight = float(config.get("loss_consistency_weight", 0.05))
        loss_kl_weight = float(config.get("loss_kl_weight", 0.05))
        loss_sisdr_weight = float(config.get("loss_sisdr_weight", 0.20))
        if parameters is not None and len(parameters) > 0:
            self.set_parameters(parameters)

        dataset = self._dataset(split, max_segments=max_segments)
        if len(dataset) == 0:
            return {"loss": 0.0, "mask_loss": 0.0, "mean_si_sdr": 0.0, "n_examples": 0}

        loader = self._loader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_mag_loss = 0.0
        total_consistency = 0.0
        total_kl = 0.0
        total_sisdr_loss = 0.0
        total_cls_loss = 0.0
        total_cls_correct = 0.0
        total_si_sdr = {stem: [] for stem in stem_names()}
        total_si_sdr_mix = {stem: [] for stem in stem_names()}
        steps = 0

        with torch.inference_mode():
            for batch in loader:
                mix_mag = batch["mix_mag"].to(self.device)
                mix_mag_raw = batch["mix_mag_raw"].to(self.device)
                mix_mag_raw_np = mix_mag_raw.cpu().numpy()
                mix_phase = batch["mix_phase"].cpu().numpy()
                target_masks = batch["target_masks"].to(self.device)
                target_mags = batch["target_mags"].to(self.device)
                device_type = self.device.type if self.device.type == "cpu" else "cuda"
                autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
                with autocast_ctx:
                    pred_masks, cls_logits = self.model(mix_mag)
                    stem_energy = target_mags.mean(dim=(-2, -1))
                    presence_labels = (stem_energy > 1e-4).float()
                    if self.target_stem_index is not None:
                        cls_l = stem_classification_loss(cls_logits, presence_labels, self.target_stem_index)
                        cls_pred = torch.sigmoid(cls_logits[:, self.target_stem_index]) > 0.5
                        cls_acc = (cls_pred == presence_labels[:, self.target_stem_index].bool()).float().mean()
                        total_cls_loss += float(cls_l.item())
                        total_cls_correct += float(cls_acc.item())
                    est_mags = pred_masks * mix_mag_raw.unsqueeze(1)
                    if self.target_stem_index is not None:
                        idx = self.target_stem_index
                        pred_masks_sup = pred_masks[:, idx:idx+1]
                        target_masks_sup = target_masks[:, idx:idx+1]
                        est_mags_sup = est_mags[:, idx:idx+1]
                        target_mags_sup = target_mags[:, idx:idx+1]
                        consistency_w = 0.0
                    else:
                        pred_masks_sup = pred_masks
                        target_masks_sup = target_masks
                        est_mags_sup = est_mags
                        target_mags_sup = target_mags
                        consistency_w = loss_consistency_weight
                    loss, components = composite_separation_loss(
                        pred_masks_sup,
                        target_masks_sup,
                        est_mags_sup,
                        target_mags_sup,
                        mask_weight=loss_mask_weight,
                        mag_weight=loss_mag_weight,
                        consistency_weight=consistency_w,
                        kl_weight=loss_kl_weight,
                        sisdr_weight=loss_sisdr_weight,
                    )
                    mask_loss = components["mask_loss"]
                    mag_loss = components["mag_loss"]
                    consistency_loss = components["consistency_loss"]
                    kl_loss = components["kl_loss"]
                    sisdr_l = components["sisdr_loss"]
                total_loss += float(loss.item())
                total_mask_loss += float(mask_loss.item())
                total_mag_loss += float(mag_loss.item())
                total_consistency += float(consistency_loss.item())
                total_kl += float(kl_loss.item())
                total_sisdr_loss += float(sisdr_l.item())

                mixture_wave = batch["mixture_wave"].cpu().numpy()
                source_waves = batch["source_waves"]
                pred_masks_np = pred_masks.cpu().numpy()
                for batch_index in range(mix_mag_raw_np.shape[0]):
                    for stem_index, stem in enumerate(stem_names()):
                        est_mag = pred_masks_np[batch_index, stem_index] * mix_mag_raw_np[batch_index]
                        est_wave = istft_from_mag_phase(
                            est_mag,
                            mix_phase[batch_index],
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            length=mixture_wave[batch_index].shape[-1],
                        )
                        ref_wave = source_waves[stem][batch_index].cpu().numpy()
                        baseline = mixture_wave[batch_index]
                        total_si_sdr[stem].append(si_sdr(ref_wave, est_wave))
                        total_si_sdr_mix[stem].append(si_sdr(ref_wave, baseline))
                steps += 1

        mean_si_sdr = {stem: float(np.mean(scores)) if scores else 0.0 for stem, scores in total_si_sdr.items()}
        mean_si_sdr_mix = {stem: float(np.mean(scores)) if scores else 0.0 for stem, scores in total_si_sdr_mix.items()}
        si_sdr_improvement = {stem: mean_si_sdr[stem] - mean_si_sdr_mix[stem] for stem in stem_names()}

        metrics = {
            "loss": total_loss / max(1, steps),
            "mask_loss": total_mask_loss / max(1, steps),
            "mag_loss": total_mag_loss / max(1, steps),
            "consistency_loss": total_consistency / max(1, steps),
            "kl_loss": total_kl / max(1, steps),
            "sisdr_loss_spectral": total_sisdr_loss / max(1, steps),
            "mean_si_sdr": float(np.mean(list(mean_si_sdr.values()))) if mean_si_sdr else 0.0,
            "mean_si_sdr_improvement": float(np.mean(list(si_sdr_improvement.values()))) if si_sdr_improvement else 0.0,
            "cls_loss": total_cls_loss / max(1, steps),
            "cls_accuracy": total_cls_correct / max(1, steps),
            "n_examples": len(dataset),
            "target_stem": self.target_stem or "all",
        }
        self._training_history.append({
            "round": _current_round,
            "phase": "evaluate",
            "timestamp": time.time(),
            "duration_s": round(time.monotonic() - _eval_start, 1),
            "loss": metrics["loss"],
            "mean_si_sdr": metrics["mean_si_sdr"],
            "mean_si_sdr_improvement": metrics["mean_si_sdr_improvement"],
            "cls_loss": metrics["cls_loss"],
            "cls_accuracy": metrics["cls_accuracy"],
            "n_examples": metrics["n_examples"],
        })
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics

    def predict(self, parameters: List[np.ndarray] | None, inputs: list, config: dict | None = None) -> list:
        config = {} if config is None else dict(config)
        if parameters is not None:
            self.set_parameters(parameters)
        use_amp = bool(config.get("use_amp", self.use_amp)) and self.device.type == "cuda"
        chunk_duration = config.get("chunk_duration", None)

        outputs = []
        self.model.eval()
        with torch.inference_mode():
            for index, inp in enumerate(inputs):
                if isinstance(inp, str):
                    wave, _ = self._load_wave_from_path(Path(inp))
                    source_name = Path(inp).stem
                else:
                    wave = np.asarray(inp, dtype=np.float32)
                    source_name = f"sample_{index}"

                mix_mag, mix_phase = self._wave_to_input(wave)
                mix_mag_raw = self._raw_magnitude(wave)

                if chunk_duration is not None:
                    pred_masks = self._predict_chunked(mix_mag, use_amp, chunk_duration)
                else:
                    mix_tensor = torch.from_numpy(mix_mag[None, None, ...]).to(self.device)
                    device_type = self.device.type if self.device.type == "cpu" else "cuda"
                    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
                    with autocast_ctx:
                        pred_masks, _ = self.model(mix_tensor)
                        pred_masks = pred_masks.cpu().numpy()[0]

                stem_paths = {}
                target_dir = self.run_path / f"predict_{source_name}"
                target_dir.mkdir(parents=True, exist_ok=True)
                for stem_index, stem in enumerate(stem_names()):
                    est_mag = pred_masks[stem_index] * mix_mag_raw
                    est_wave = istft_from_mag_phase(
                        est_mag,
                        mix_phase,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        length=wave.shape[-1],
                    )
                    stem_path = target_dir / f"{stem}.wav"
                    self._write_wav(stem_path, est_wave)
                    stem_paths[stem] = str(stem_path)

                outputs.append({
                    "input": str(inp) if isinstance(inp, str) else source_name,
                    "output_dir": str(target_dir),
                    "stems": stem_paths,
                })
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return outputs

    def get_training_status(self) -> dict:
        return {
            "busy": self.lock.locked(),
            "current_round": self._round,
            "target_stem": self.target_stem,
            "model_name": self.model_name,
            "history": list(self._training_history),
        }

    def in_usage(self, value: int | None = None):
        if value is None:
            return self.lock.locked()
        if value == 1:
            if not self.lock.locked():
                self.lock.acquire()
        else:
            if self.lock.locked():
                try:
                    self.lock.release()
                except RuntimeError:
                    pass

    def set_run_path(self, run_path: str):
        self.run_path = Path(run_path)
        self.run_path.mkdir(parents=True, exist_ok=True)

    def _dataset(self, split: str, max_segments: int | None = None) -> MusdbSeparationDataset:
        cache_key = (
            f"{split}:{max_segments}:{self.client_id}:{self.num_clients}:"
            f"{self.augment_train}:{self.augment_gain_db}:{self.augment_noise_std}:"
            f"{self.augment_stem_dropout_prob}:{self.augment_shift_seconds}:"
            f"{self.augment_remix_prob}:{self.augment_phase_flip}"
        )
        if cache_key not in self._cache:
            self._cache[cache_key] = MusdbSeparationDataset(
                self.data_root,
                split=split,
                target_sr=self.sr,
                segment_seconds=self.segment_seconds,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                max_segments=max_segments,
                client_id=self.client_id,
                num_clients=self.num_clients,
                augment_train=self.augment_train if split == "train" else False,
                augment_gain_db=self.augment_gain_db,
                augment_noise_std=self.augment_noise_std,
                augment_stem_dropout_prob=self.augment_stem_dropout_prob,
                augment_shift_seconds=self.augment_shift_seconds,
                augment_remix_prob=self.augment_remix_prob,
                augment_phase_flip=self.augment_phase_flip,
            )
        return self._cache[cache_key]

    def split_track_names(self, split: str, max_segments: int | None = None) -> set[str]:
        dataset = self._dataset(split, max_segments=max_segments)
        return dataset.track_names()

    def update_scheduler(self, metric: float | None = None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                return
            self.scheduler.step(metric)

    def _configure_scheduler(self, config: dict, learning_rate: float) -> None:
        scheduler_name = str(config.get("scheduler", "none")).lower()
        signature = (
            scheduler_name,
            float(learning_rate),
            int(config.get("scheduler_step_size", 5)),
            float(config.get("scheduler_gamma", 0.5)),
            int(config.get("scheduler_t_max", 10)),
            float(config.get("scheduler_min_lr", 1e-6)),
            str(config.get("scheduler_mode", "max")).lower(),
        )

        if scheduler_name == "none":
            self.scheduler = None
            self._scheduler_signature = signature
            self.optimizer.param_groups[0]["lr"] = learning_rate
            return

        if self._scheduler_signature == signature and self.scheduler is not None:
            return

        self.optimizer.param_groups[0]["lr"] = learning_rate
        if scheduler_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=max(1, int(config.get("scheduler_step_size", 5))),
                gamma=float(config.get("scheduler_gamma", 0.5)),
            )
        elif scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(config.get("scheduler_t_max", 10))),
                eta_min=float(config.get("scheduler_min_lr", 1e-6)),
            )
        elif scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=str(config.get("scheduler_mode", "max")).lower(),
                factor=float(config.get("scheduler_gamma", 0.5)),
                patience=max(1, int(config.get("scheduler_step_size", 2))),
                min_lr=float(config.get("scheduler_min_lr", 1e-6)),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: none, step, cosine, plateau")
        self._scheduler_signature = signature

    @staticmethod
    def _collate_batch(batch: List[dict]) -> dict:
        """
        Collate a batch of examples, padding tensor entries on the time (last) dimension
        when necessary so tensors with variable length can be stacked. This keeps
        backward compatibility with fixed-length segments while allowing full-track
        examples (variable-length) to be batched together.

        Strategy:
        - For tensor values: if all shapes match -> stack directly.
          If shapes differ, pad each tensor on the last dimension to the maximum
          last-dimension size across the batch using a right-side pad (value=0),
          then stack.
        - For nested dict values: apply the same logic per nested key.
        - Non-tensor values are collected into lists as before.
        """
        result: Dict[str, object] = {}
        keys = batch[0].keys()
        for key in keys:
            values = [item[key] for item in batch]
            # Handle tensors (possibly variable-length in last dim)
            if isinstance(values[0], torch.Tensor):
                shapes = [tuple(v.shape) for v in values]
                if all(s == shapes[0] for s in shapes):
                    result[key] = torch.stack(values, dim=0)
                else:
                    # Pad along last dimension to match the maximum length
                    max_last = max(s[-1] for s in shapes)
                    padded = []
                    for v in values:
                        if v.shape[-1] == max_last:
                            padded.append(v)
                        else:
                            pad_amount = max_last - v.shape[-1]
                            # pad tuple is (left, right) pairs for last dims; (0, pad_amount)
                            v_padded = F.pad(v, (0, pad_amount))
                            padded.append(v_padded)
                    result[key] = torch.stack(padded, dim=0)
            elif isinstance(values[0], dict):
                nested: Dict[str, object] = {}
                nested_keys = values[0].keys()
                for nested_key in nested_keys:
                    nested_values = [value[nested_key] for value in values]
                    if isinstance(nested_values[0], torch.Tensor):
                        shapes = [tuple(v.shape) for v in nested_values]
                        if all(s == shapes[0] for s in shapes):
                            nested[nested_key] = torch.stack(nested_values, dim=0)
                        else:
                            max_last = max(s[-1] for s in shapes)
                            padded = []
                            for v in nested_values:
                                if v.shape[-1] == max_last:
                                    padded.append(v)
                                else:
                                    pad_amount = max_last - v.shape[-1]
                                    v_padded = F.pad(v, (0, pad_amount))
                                    padded.append(v_padded)
                            nested[nested_key] = torch.stack(padded, dim=0)
                    else:
                        nested[nested_key] = nested_values
                result[key] = nested
            else:
                result[key] = values
        return result

    def _loader(self, dataset: MusdbSeparationDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_batch,
        )

    def _wave_to_input(self, wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from .utils.audio import normalize_magnitude, stft_mag_phase

        if wave.ndim > 1:
            wave = np.mean(wave, axis=1)
        magnitude, phase = stft_mag_phase(wave, n_fft=self.n_fft, hop_length=self.hop_length)
        return normalize_magnitude(magnitude, log_scale=True), phase

    def _raw_magnitude(self, wave: np.ndarray) -> np.ndarray:
        from .utils.audio import stft_mag_phase

        if wave.ndim > 1:
            wave = np.mean(wave, axis=1)
        magnitude, _ = stft_mag_phase(wave, n_fft=self.n_fft, hop_length=self.hop_length)
        return magnitude

    def _load_wave_from_path(self, path: Path) -> tuple[np.ndarray, int]:
        from .utils.audio import load_audio_segment

        wave, sr = load_audio_segment(str(path), target_sr=self.sr, mono=True)
        return wave, sr

    def _write_wav(self, path: Path, wave: np.ndarray) -> None:
        import soundfile as sf

        sf.write(str(path), wave.astype(np.float32, copy=False), self.sr)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.run_path / filename
        torch.save(self.model.state_dict(), path)

    def save_checkpoint(self, filename: str) -> Path:
        path = self.run_path / filename
        torch.save(self.model.state_dict(), path)
        return path

    def _predict_chunked(self, mix_mag: np.ndarray, use_amp: bool, chunk_duration: float) -> np.ndarray:
        """Process spectrogram in overlapping chunks to reduce peak memory usage."""
        frames_per_second = self.sr / self.hop_length
        chunk_frames = int(chunk_duration * frames_per_second)
        overlap_frames = chunk_frames // 4  # 25% overlap for smooth boundary blending

        time_frames = mix_mag.shape[1]

        if time_frames <= chunk_frames:
            mix_tensor = torch.from_numpy(mix_mag[None, None, ...]).to(self.device)
            device_type = self.device.type if self.device.type == "cpu" else "cuda"
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
            with autocast_ctx:
                pred_masks, _ = self.model(mix_tensor)
                pred_masks = pred_masks.cpu().numpy()[0]
            return pred_masks

        num_stems = 4
        result_masks = np.zeros((num_stems, mix_mag.shape[0], time_frames), dtype=np.float32)
        blend_weights = np.zeros((time_frames,), dtype=np.float32)

        start = 0
        while start < time_frames:
            end = min(start + chunk_frames, time_frames)
            chunk_mag = mix_mag[:, start:end]

            if chunk_mag.shape[1] < chunk_frames:
                pad_width = chunk_frames - chunk_mag.shape[1]
                chunk_mag = np.pad(chunk_mag, ((0, 0), (0, pad_width)), mode="reflect")

            mix_tensor = torch.from_numpy(chunk_mag[None, None, ...]).to(self.device)
            device_type = self.device.type if self.device.type == "cpu" else "cuda"
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
            with autocast_ctx:
                chunk_masks, _ = self.model(mix_tensor)
                chunk_masks = chunk_masks.cpu().numpy()[0]

            if chunk_mag.shape[1] > (end - start):
                chunk_masks = chunk_masks[:, :, :end - start]

            window = np.hanning(chunk_masks.shape[2] * 2)[chunk_masks.shape[2]:]

            chunk_end = min(start + chunk_masks.shape[2], time_frames)
            chunk_len = chunk_end - start
            result_masks[:, :, start:chunk_end] += chunk_masks[:, :, :chunk_len] * window[:chunk_len]
            blend_weights[start:chunk_end] += window[:chunk_len]

            start += chunk_frames - overlap_frames
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        blend_weights[blend_weights == 0] = 1.0
        result_masks = result_masks / blend_weights[None, None, :]

        return result_masks

    def classify(self, audio_bytes: bytes) -> dict:
        """Return stem presence classification for this client's assigned stem.

        Returns a dict with keys: stem, present, confidence.
        If no target stem is configured, all four stems are returned.
        """
        import io
        import soundfile as sf
        from .utils.audio import normalize_magnitude, stft_mag_phase

        wave, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        # Truncate to 5 s — the GAP bottleneck discards position anyway, so
        # extra duration just wastes compute and risks OOM on long inputs.
        max_samples = 5 * self.sr
        if wave.shape[0] > max_samples:
            wave = wave[:max_samples]

        mag, _ = stft_mag_phase(wave, n_fft=self.n_fft, hop_length=self.hop_length)
        log_mag, _ = normalize_magnitude(mag, log_scale=True), None
        mix_tensor = torch.from_numpy(log_mag[None, None, ...]).to(self.device)

        self.model.eval()
        device_type = self.device.type if self.device.type == "cpu" else "cuda"
        use_amp = _env_bool("USE_AMP")
        autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
        with torch.no_grad(), autocast_ctx:
            _, cls_logits = self.model(mix_tensor)

        probs = torch.sigmoid(cls_logits[0]).cpu().tolist()
        stems = list(stem_names())

        if self.target_stem_index is not None:
            idx = self.target_stem_index
            conf = float(probs[idx])
            return {"stem": stems[idx], "present": conf >= 0.5, "confidence": conf}

        return {
            "stems": {s: {"present": probs[i] >= 0.5, "confidence": float(probs[i])} for i, s in enumerate(stems)}
        }

    def _chunk_frames_from_duration(self, duration_seconds: float | int | None) -> int | None:
        if duration_seconds is None:
            return None
        duration = float(duration_seconds)
        if duration <= 0:
            raise ValueError("train_chunk_duration must be > 0 when provided.")
        frames_per_second = self.sr / self.hop_length
        return max(1, int(duration * frames_per_second))

    @staticmethod
    def _build_chunk_ranges(total_frames: int, chunk_frames: int | None, overlap_ratio: float = 0.0) -> List[Tuple[int, int]]:
        if total_frames <= 0:
            return [(0, 0)]
        if chunk_frames is None or chunk_frames >= total_frames:
            return [(0, total_frames)]

        overlap_frames = int(chunk_frames * overlap_ratio)
        stride = max(1, chunk_frames - overlap_frames)
        ranges: List[Tuple[int, int]] = []
        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            ranges.append((start, end))
            if end >= total_frames:
                break
            start += stride
        return ranges

def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except ValueError:
        return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    app = MLApp()
    print(app.get_data({"max_segments": 3}))
