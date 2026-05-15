from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        groups = _group_count(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
            ResidualBlock(out_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [0, max(0, diff_w), 0, max(0, diff_h)])
            x = x[..., : skip.shape[-2], : skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetSmall(nn.Module):
    """Residual spectrogram U-Net that predicts four separation probability masks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4, base_filters: int = 24, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters

        self.stem = ResidualBlock(in_channels, base_filters, dropout=0.0)
        self.down1 = DownBlock(base_filters, base_filters * 2, dropout=dropout)
        self.down2 = DownBlock(base_filters * 2, base_filters * 4, dropout=dropout)
        self.down3 = DownBlock(base_filters * 4, base_filters * 8, dropout=dropout)
        self.down4 = DownBlock(base_filters * 8, base_filters * 12, dropout=dropout)
        self.bottleneck = ResidualBlock(base_filters * 12, base_filters * 12, dropout=dropout)
        self.up4 = UpBlock(base_filters * 12, base_filters * 8, base_filters * 8, dropout=dropout)
        self.up3 = UpBlock(base_filters * 8, base_filters * 4, base_filters * 4, dropout=dropout)
        self.up2 = UpBlock(base_filters * 4, base_filters * 2, base_filters * 2, dropout=dropout)
        self.up1 = UpBlock(base_filters * 2, base_filters, base_filters, dropout=dropout)
        self.head = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = x.shape[-2], x.shape[-1]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = pad_hw
        if pad_h:
            x = x[..., :-pad_h, :]
        if pad_w:
            x = x[..., :, :-pad_w]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, freq, time), got {tuple(x.shape)}")

        x, pad_hw = self._pad_to_multiple(x, multiple=16)
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        b = self.bottleneck(s5)
        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.head(x)
        x = torch.softmax(x, dim=1)
        x = self._crop(x, pad_hw)
        return x


if __name__ == "__main__":
    model = UNetSmall()
    dummy = torch.randn(2, 1, 257, 251)
    out = model(dummy)
    print(out.shape)
