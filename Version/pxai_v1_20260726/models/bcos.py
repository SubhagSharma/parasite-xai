"""B-cos head (Böhle et al. 2022) — alignment-based weight-faithful explanation.

A B-cos transform raises the cosine alignment between weights and input to a
power B, so a trained B-cos model collapses to a single input-dependent linear
map W(x) whose rows ARE an exact, holistic explanation (no approximation gap).
Here we provide a lightweight B-cos classification head that attaches to the
backbone feature map; the explanation is the dynamic linear contribution map.

This is a compact head-only variant. For a fully weight-faithful pipeline the
backbone convs should also be B-cos; that is an A2/A7 extension.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BcosConv2d(nn.Module):
    """1x1 B-cos conv: scales output by |cos(x, w)|^(B-1)."""

    def __init__(self, in_ch, out_ch, b: float = 2.0):
        super().__init__()
        self.lin = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.b = b

    def forward(self, x):
        w = self.lin.weight                                   # (O, I, 1, 1)
        out = self.lin(x)                                     # (B,O,H,W)
        wn = w.flatten(1).norm(dim=1).clamp_min(1e-6)         # (O,)
        xn = x.norm(dim=1, keepdim=True).clamp_min(1e-6)      # (B,1,H,W)
        cos = out / (wn.view(1, -1, 1, 1) * xn)
        return out * cos.abs().pow(self.b - 1)


class BcosHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, b: float = 2.0):
        super().__init__()
        self.block = BcosConv2d(in_channels, num_classes, b=b)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat):
        return self.pool(self.block(feat)).flatten(1)

    @torch.no_grad()
    def explain(self, feat):
        """Spatial class-contribution map (B, num_classes, H, W)."""
        return {"contrib_map": self.block(feat)}
