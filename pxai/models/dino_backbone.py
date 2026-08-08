"""Frozen DINOv2 backbone exposing pxai's interface. NEW FILE."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoBackbone(nn.Module):
    """Frozen DINOv2 ViT, reshaped to (B, C, H, W) so existing heads attach unchanged.

    FROZEN deliberately: fine-tuning would let the classification objective collapse the
    within-class rank this experiment exists to preserve. Only the head trains.

    grid_out: average-pool the patch grid to this size before returning. DINOv2 patch 14
    gives 16x16 on a 224 input against MobileViT's 7x7, so `grid_out=7` matches the
    supervised setting exactly and removes grid size as a confound.
    """

    def __init__(self, name="vit_small_patch14_dinov2.lvd142m", grid_out=None):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=True, num_classes=0,
                                     img_size=224)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()
        self.grid_out = grid_out
        with torch.no_grad():
            self.out_channels = int(self.forward(torch.zeros(1, 3, 224, 224)).shape[1])

    def train(self, mode=True):
        # stay in eval: frozen weights, and DINOv2 has no dropout worth toggling
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            t = self.net.forward_features(x)[:, 1:]      # drop CLS
        n = int(round(math.sqrt(t.shape[1])))
        f = t.transpose(1, 2).reshape(t.shape[0], -1, n, n)
        if self.grid_out and n != self.grid_out:
            f = F.adaptive_avg_pool2d(f, self.grid_out)
        return f.detach()
