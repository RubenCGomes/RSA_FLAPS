from __future__ import annotations

import torch

from ml_app.models.unet_small import UNetSmall


def test_unet_output_shape_matches_four_stems():
    model = UNetSmall(in_channels=1, out_channels=4, base_filters=24)
    x = torch.randn(2, 1, 257, 251)
    y = model(x)
    assert y.shape == (2, 4, 257, 251)
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)
    assert torch.allclose(y.sum(dim=1).mean(), torch.tensor(1.0), atol=1e-4)
