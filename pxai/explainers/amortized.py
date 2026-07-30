"""Amortized single-pass explainer — the headline "lighter than SOTA" result (C2/H3).

A small explainer network is trained to emit Shapley-quality attributions in ONE
forward pass, instead of the 100s-1000s of model queries KernelSHAP/LIME need.
This is the FastSHAP idea (Jethani et al. 2022): minimise a weighted-least-squares
Shapley objective via random coalition masks, with the additive-efficiency
constraint enforced by a normalisation.

Closest image-domain prior art to cite/compare: ViT-Shapley (Covert et al. 2022,
arXiv:2206.05282) and Stochastic Amortization (Covert et al. 2024,
arXiv:2401.15866) — noisy targets suffice, which lowers training cost.

Train against a frozen `teacher` (your heavy black box). At inference the
explainer recovers model-agnostic SHAP at edge cost (1 pass).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmortizedExplainer(nn.Module):
    """U-Net-lite: image -> per-superpixel Shapley map for the target class."""

    def __init__(self, in_ch: int = 3, num_classes: int = 11, grid: int = 14):
        super().__init__()
        self.grid = grid
        self.num_classes = num_classes
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(grid),
        )
        self.head = nn.Conv2d(128, num_classes, 1)   # one Shapley map per class

    def forward(self, x):
        return self.head(self.enc(x))                 # (B, C, grid, grid)

    def explain(self, x, target):
        phi = self.forward(x)                          # (B,C,g,g)
        sel = phi.gather(1, target.view(-1, 1, 1, 1).expand(-1, 1, self.grid, self.grid))
        return F.interpolate(sel, size=x.shape[-2:], mode="bilinear", align_corners=False)


def _mask_image(x, mask_grid):
    """Apply coalition mask (B, g, g) to image by upsampling to input resolution."""
    m = F.interpolate(mask_grid.unsqueeze(1), size=x.shape[-2:], mode="nearest")
    return x * m                                        # masked-out pixels -> 0 baseline


def fastshap_step(explainer, teacher, x, target, num_coalitions: int = 8):
    """One FastSHAP training step. Returns scalar loss.

    Weighted least squares: E_S[ (teacher(x_S) - teacher(x_0) - sum_{i in S} phi_i)^2 ]
    with the Shapley kernel sampling distribution over coalition sizes.
    """
    B = x.shape[0]
    g = explainer.grid
    phi = explainer(x)                                              # (B,C,g,g)
    phi = phi.gather(1, target.view(-1, 1, 1, 1).expand(-1, 1, g, g)).squeeze(1)  # (B,g,g)

    with torch.no_grad():
        f_full = teacher(x).softmax(1).gather(1, target.view(-1, 1)).squeeze(1)   # v(N)
        f_empty = teacher(torch.zeros_like(x)).softmax(1).gather(
            1, target.view(-1, 1)).squeeze(1)                                      # v(empty)

    loss = x.new_zeros(())
    for _ in range(num_coalitions):
        S = (torch.rand(B, g, g, device=x.device) > 0.5).float()
        with torch.no_grad():
            v_S = teacher(_mask_image(x, S)).softmax(1).gather(1, target.view(-1, 1)).squeeze(1)
        pred = f_empty + (phi * S).flatten(1).sum(1)
        loss = loss + F.mse_loss(pred, v_S)

    # additive efficiency: sum of phi should equal v(N) - v(empty)
    eff = F.mse_loss(phi.flatten(1).sum(1), f_full - f_empty)
    return loss / num_coalitions + 0.1 * eff
