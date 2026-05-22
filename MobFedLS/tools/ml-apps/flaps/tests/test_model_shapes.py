from __future__ import annotations

import torch

from ml_app.models.unet_large import UNetLarge
from ml_app.models.unet_small import UNetSmall


def test_unet_small_mask_shape():
    model = UNetSmall(in_channels=1, out_channels=4, base_filters=24)
    x = torch.randn(2, 1, 257, 251)
    masks, cls_logits = model(x)
    assert masks.shape == (2, 4, 257, 251)
    assert torch.all(masks >= 0)
    assert torch.all(masks <= 1)
    assert torch.allclose(masks.sum(dim=1).mean(), torch.tensor(1.0), atol=1e-4)


def test_unet_small_cls_logits_shape():
    model = UNetSmall(in_channels=1, out_channels=4, base_filters=24)
    x = torch.randn(2, 1, 257, 251)
    _, cls_logits = model(x)
    assert cls_logits.shape == (2, 4)


def test_unet_large_dual_head_shapes():
    model = UNetLarge(in_channels=1, out_channels=4, base_filters=24)
    x = torch.randn(2, 1, 257, 251)
    masks, cls_logits = model(x)
    assert masks.shape == (2, 4, 257, 251)
    assert cls_logits.shape == (2, 4)
    assert torch.allclose(masks.sum(dim=1).mean(), torch.tensor(1.0), atol=1e-4)


def test_unet_small_non_square_input():
    """Verify padding/crop handles inputs not divisible by 16."""
    model = UNetSmall(in_channels=1, out_channels=4, base_filters=8)
    x = torch.randn(1, 1, 129, 100)
    masks, cls_logits = model(x)
    assert masks.shape == (1, 4, 129, 100)
    assert cls_logits.shape == (1, 4)
