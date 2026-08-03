"""Lightweight feature-extractor backbones (A1 ablation).

All backbones return a spatial feature map (B, C, H, W) so prototype / B-cos /
Grad-CAM heads can attach. We use timm `features_only` to get the last stage.

Footprint targets (the "sub-10 MB" claim) — fp32 param sizes are approximate:
    mobilevit_xs      ~2.3 M params   (~9 MB)   <- default
    efficientnet_lite0~4.6 M params   (~18 MB, prune/quantize for <10)
    ghostnet_100      ~5.2 M params
    convnext_tiny     ~28 M params    (heavy black-box reference, NOT lightweight)
"""
from __future__ import annotations

import timm
import torch.nn as nn

_SUPPORTED = {
    "mobilevit_xs": "mobilevit_xs",
    "mobilevit_xxs": "mobilevit_xxs",
    "efficientnet_lite0": "efficientnet_lite0",
    "ghostnet_100": "ghostnet_100",
    "convnext_tiny": "convnext_tiny",   # heavy reference
}


class Backbone(nn.Module):
    """Wraps a timm model to expose a single last-stage feature map + its channel dim."""

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        if name not in _SUPPORTED:
            raise ValueError(f"backbone {name!r} not in {list(_SUPPORTED)}")
        self.net = timm.create_model(
            _SUPPORTED[name], pretrained=pretrained,
            features_only=True)
        self.out_channels = self.net.feature_info.channels()[-1]
        self.name = name

    def forward(self, x):
        return self.net(x)[-1]            # (B, C, H, W)


def build_backbone(cfg) -> Backbone:
    b = cfg["backbone"]
    return Backbone(b["name"], b["pretrained"])
