from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from ml_app.dataset.musdb_loader import MusdbSeparationDataset


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_dataset_loads_real_musdb_sample():
    dataset_root = repo_root() / "assets" / "datasets" / "musdb18hq"
    assert dataset_root.exists(), "musdb18hq dataset is required for this smoke test"

    dataset = MusdbSeparationDataset(
        dataset_root,
        split="train",
        target_sr=16000,
        segment_seconds=5.0,
        max_segments=2,
    )

    assert len(dataset) == 2
    sample = dataset[0]
    mix_mag = cast(torch.Tensor, sample["mix_mag"])
    target_masks = cast(torch.Tensor, sample["target_masks"])
    target_mags = cast(torch.Tensor, sample["target_mags"])
    mix_mag_raw = cast(torch.Tensor, sample["mix_mag_raw"])

    assert mix_mag.shape[0] == 1
    assert target_masks.shape[0] == 4
    assert target_mags.shape[0] == 4
    assert torch.all(target_masks >= 0)

    mask_sum = target_masks.sum(dim=0)
    active_bins = mix_mag_raw > 0
    assert bool(active_bins.any())
    assert torch.allclose(mask_sum[active_bins].mean(), torch.tensor(1.0), atol=0.25)
