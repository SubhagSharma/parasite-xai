"""Heavy black-box classifier — the baseline to BEAT on faithfulness/cost.

This is the model the post-hoc explainers (Grad-CAM, IG, LIME, KernelSHAP) and
the amortized FastSHAP explainer are applied to. Use a heavier backbone
(convnext_tiny) so the comparison is honest: we want to show an inherently
interpretable sub-10 MB model is MORE faithful than post-hoc saliency on a
stronger black box.
"""
from __future__ import annotations

import torch.nn as nn
from ..backbones import Backbone


class BlackBox(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "convnext_tiny",
                 pretrained: bool = True):
        super().__init__()
        self.backbone = Backbone(backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.fc(self.pool(f).flatten(1))
