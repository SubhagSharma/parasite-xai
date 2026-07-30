"""Prototype head — ProtoPNet / PIP-Net flavour ("this looks like that").

The forward pass IS the explanation: each class score is a sum of similarities
to learned prototypical patches, so faithfulness is by construction (Rudin's
argument). Set `pip_sparsity=True` for a PIP-Net-style sparse, non-negative
scoring sheet that can abstain (all-low similarities -> low max class score).

Explanation API: `explain(x)` returns per-prototype similarity maps you can
upsample to the input for "this region looks like prototype p of class c".

This is a compact, didactic re-implementation suitable for a lightweight
backbone; cite Chen et al. 2019 (ProtoPNet) and Nauta et al. 2023 (PIP-Net).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 protos_per_class: int = 5, proto_dim: int = 128,
                 pip_sparsity: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.ppc = protos_per_class
        self.num_protos = num_classes * protos_per_class
        self.proto_dim = proto_dim
        self.pip_sparsity = pip_sparsity

        # 1x1 add-on conv maps backbone features -> prototype embedding space
        self.add_on = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, 1), nn.ReLU(),
            nn.Conv2d(proto_dim, proto_dim, 1), nn.Sigmoid(),
        )
        # prototype vectors (P, D, 1, 1)
        self.prototypes = nn.Parameter(torch.rand(self.num_protos, proto_dim, 1, 1))

        # fixed class-identity map: prototype p belongs to class p // ppc
        ident = torch.zeros(self.num_protos, num_classes)
        for p in range(self.num_protos):
            ident[p, p // protos_per_class] = 1.0
        self.register_buffer("proto_class", ident)

        # last layer: prototypes -> class logits.
        self.last = nn.Linear(self.num_protos, num_classes, bias=False)
        with torch.no_grad():
            # init so each prototype positively votes its own class (PIP-Net: non-neg)
            self.last.weight.copy_(ident.t())

    def _similarities(self, x):
        """Return (sim_pooled (B,P), sim_maps (B,P,H,W)) using L2 prototype distance."""
        z = self.add_on(x)                                   # (B, D, H, W)
        B, D, H, W = z.shape
        z2 = (z ** 2).sum(1, keepdim=True)                   # (B,1,H,W)
        p = self.prototypes                                  # (P,D,1,1)
        p2 = (p ** 2).sum(1).view(1, -1, 1, 1)               # (1,P,1,1)
        zp = F.conv2d(z, p)                                  # (B,P,H,W)
        dist = F.relu(z2 + p2 - 2 * zp)                      # squared L2, >=0
        sim = torch.log((dist + 1.0) / (dist + 1e-4))        # ProtoPNet similarity
        sim_pooled = F.max_pool2d(sim, sim.shape[-2:]).flatten(1)  # (B,P)
        return sim_pooled, sim

    def forward(self, feat):
        sim_pooled, _ = self._similarities(feat)
        if self.pip_sparsity:
            # PIP-Net: non-negative last layer + sparsity -> abstaining scoring sheet
            w = F.relu(self.last.weight)
            logits = F.linear(sim_pooled, w)
        else:
            logits = self.last(sim_pooled)
        return logits

    @torch.no_grad()
    def explain(self, feat):
        """Return dict with per-prototype pooled similarity and spatial maps."""
        sim_pooled, sim_maps = self._similarities(feat)
        return {"sim_pooled": sim_pooled, "sim_maps": sim_maps,
                "proto_class": self.proto_class}

    # --- regularizers used by train.py ---
    def cluster_sep_cost(self, feat, targets):
        """ProtoPNet cluster (pull correct-class protos close) + separation cost."""
        sim_pooled, _ = self._similarities(feat)             # (B,P) higher = closer
        oh = self.proto_class[:, targets].t()                # (B,P) 1 if proto's class==target
        cluster = -(sim_pooled * oh).max(1).values.mean()    # maximise best same-class sim
        sep = (sim_pooled * (1 - oh)).max(1).values.mean()   # minimise best other-class sim
        return cluster, sep

    def l1_last(self):
        """Sparsity on cross-class connections (PIP-Net compactness)."""
        off = (1 - self.proto_class.t())                     # (C,P)
        return (self.last.weight * off).abs().sum()
