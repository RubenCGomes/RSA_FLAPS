from __future__ import annotations

import torch

from .unet_small import UNetSmall


class UNetLarge(UNetSmall):
    """Wider variant of the base U-Net for higher separation capacity."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4, base_filters: int = 24, dropout: float = 0.15):
        effective_filters = max(24, int(base_filters))
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_filters=effective_filters, dropout=dropout)


if __name__ == "__main__":
    model = UNetLarge()
    dummy = torch.randn(2, 1, 257, 251)
    masks, cls_logits = model(dummy)
    print("masks:", masks.shape, "cls_logits:", cls_logits.shape)

