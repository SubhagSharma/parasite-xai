"""Multi-scale prototype head — prototypes that can see structures smaller than an egg.

===============================================================================
THE PROBLEM THIS SOLVES
===============================================================================
Part II recovered localisation to conc_+ ~ 5.5 against a ceiling of 16.3. The limit is
the effective RECEPTIVE FIELD of the stride-32 stage, measured at ~100 x 100 px
(MECHANISTIC_MODEL.md §3). What that buys and what it does not:

    structure            frame %   px side    within a 100px RF?
    whole egg (mean)       11.8%      77       yes
    shell wall              2.0%      32       no
    operculum / plug        0.6%      17       no
    cilia / striation       0.3%      12       no

So the explanation can say "the egg" and cannot say "the polar plug". No loss term
fixes this: the diversity work in protopnet_diverse.py makes prototypes use different
CHANNELS, which is orthogonal to the scale limit. Two prototypes with disjoint channels
still produce ~100 px blobs.

===============================================================================
THE FIX: ATTACH PROTOTYPES AT MORE THAN ONE DEPTH
===============================================================================
The backbone already emits a feature pyramid; only the last map is currently used.

    stage   stride   cell    grid     finest structure expressible
      1        4      4px    56x56             ~4 px
      2        8      8px    28x28             ~8 px      <- cilia, plugs
      3       16     16px    14x14            ~16 px      <- shell wall
      4/5     32     32px     7x7             ~32 px      <- whole-egg shape

Standard ProtoPNet attaches all prototypes to the last stage. This head splits them
across stages, so a class gets both FINE prototypes (small RF, sees texture and small
features) and COARSE ones (large RF, sees global shape).

TWO PROBLEMS SOLVED AT ONCE
  1. SCALE. Fine-stage prototypes have receptive fields of order 8-32 px, which is the
     scale of the structures that were previously unreachable.
  2. REDUNDANCY. Prototypes at different stages CANNOT be redundant -- they physically
     cannot see the same thing. The measured collapse (effective rank 1.02 of 5,
     cosine 0.988) is between prototypes that all live at stride 32. Cross-stage
     prototypes are diverse by construction, not by penalty.

PREDICTION, RECORDED BEFORE THE RUN
  Using the receptive-field model conc_grad ~ 1/rho: a stride-8 prototype should reach
  conc_+ well above the stride-32 ceiling of ~5.5, plausibly 10+ on small-egg classes,
  because its RF is a fraction of the object rather than several times it.
  If fine-stage prototypes score no better than coarse ones, the RF model is wrong.

===============================================================================
WHY THIS IS NOT A DROP-IN
===============================================================================
`model.features(x)` returns ONE tensor and every head, probe and the push routine
assumes that. This file therefore ships three things:

    MultiScaleProtoHead    the head, taking a LIST of feature maps
    pyramid_forward()      helper returning the full pyramid from a timm backbone
    push_multiscale()      per-stage projection, since the existing push handles one map

`apply_multiscale_head.py` wires them in behind `kind: protopnet_ms`, leaving
`protopnet` and `protopnet_diverse` untouched.

===============================================================================
CONFIG
===============================================================================
    model:
      kind: protopnet_ms
      protopnet_ms:
        stages: [1, 3]                 # indices into the feature pyramid
        protos_per_class_per_stage: [2, 3]
        proto_dim: 128
        pip_sparsity: true

  stages [1, 3] on mobilevit_xs is stride 8 and stride 32 -- one fine, one coarse.
  Use [0, 1, 3] for three scales; keep the total prototypes per class at 5 so the
  comparison against the single-scale baseline is like-for-like.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def pyramid_forward(backbone, x) -> List[torch.Tensor]:
    """Return every feature map, not just the last.

    pxai's backbone wrapper does `self.net(x)[-1]`. The underlying timm model with
    features_only=True already computes the whole pyramid, so this costs nothing extra.
    """
    net = getattr(backbone, "net", backbone)
    out = net(x)
    return list(out) if isinstance(out, (list, tuple)) else [out]


class MultiScaleProtoHead(nn.Module):
    """ProtoPNet head with prototypes distributed over several backbone stages."""

    def __init__(self, stage_channels: Sequence[int], num_classes: int,
                 stages: Sequence[int] = (1, 3),
                 protos_per_class_per_stage: Sequence[int] = (2, 3),
                 proto_dim: int = 128, pip_sparsity: bool = True):
        super().__init__()
        assert len(stages) == len(protos_per_class_per_stage), \
            "stages and protos_per_class_per_stage must have the same length"
        self.stages = list(stages)
        self.ppcs = list(protos_per_class_per_stage)
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.pip_sparsity = pip_sparsity
        self.ppc = sum(self.ppcs)                     # total per class, all stages
        self.num_protos = self.P = num_classes * self.ppc

        # one add_on per stage: channel counts differ down the pyramid
        self.add_ons = nn.ModuleList([
            nn.Sequential(nn.Conv2d(stage_channels[s], proto_dim, 1), nn.ReLU(),
                          nn.Conv2d(proto_dim, proto_dim, 1), nn.Sigmoid())
            for s in self.stages])
        self.prototypes = nn.Parameter(torch.rand(self.P, proto_dim, 1, 1))

        # prototype p belongs to (class, stage). Layout is class-major so that
        # p // ppc == class, matching the single-scale head and every existing probe.
        pc = torch.zeros(self.P, num_classes)
        stage_of = torch.zeros(self.P, dtype=torch.long)
        for c in range(num_classes):
            off = c * self.ppc
            k = 0
            for si, n in enumerate(self.ppcs):
                for _ in range(n):
                    pc[off + k, c] = 1.0
                    stage_of[off + k] = si
                    k += 1
        self.register_buffer("proto_class", pc)
        self.register_buffer("stage_of", stage_of)

        self.last = nn.Linear(self.P, num_classes, bias=False)
        with torch.no_grad():
            self.last.weight.copy_(pc.t())

    # ------------------------------------------------------------------ similarity
    def _similarities(self, feats: List[torch.Tensor]):
        """-> pooled (B,P), and a list of per-stage similarity maps.

        Each prototype is compared only against the stage it belongs to, so a
        stride-8 prototype never sees stride-32 features.
        """
        B = feats[0].shape[0]
        pooled = feats[0].new_zeros(B, self.P)
        maps = []
        for si, s in enumerate(self.stages):
            z = self.add_ons[si](feats[s])                       # (B,D,H,W)
            _, D, H, W = z.shape
            idx = (self.stage_of == si).nonzero(as_tuple=True)[0]
            pv = self.prototypes[idx].view(len(idx), D)
            zf = z.permute(0, 2, 3, 1).reshape(-1, D)
            d = (zf ** 2).sum(1, keepdim=True) - 2 * zf @ pv.t() + (pv ** 2).sum(1)
            d = d.clamp_min(0).view(B, H, W, len(idx)).permute(0, 3, 1, 2)
            sim = torch.log((d + 1.0) / (d + 1e-4))              # (B,k,H,W)
            pooled[:, idx] = F.max_pool2d(sim, sim.shape[-2:]).flatten(1)
            maps.append(sim)
        return pooled, maps

    def forward(self, feats):
        pooled, _ = self._similarities(feats)
        w = F.relu(self.last.weight) if self.pip_sparsity else self.last.weight
        return F.linear(pooled, w)

    @torch.no_grad()
    def explain(self, feats):
        pooled, maps = self._similarities(feats)
        return {"sim_pooled": pooled, "sim_maps_per_stage": maps,
                "proto_class": self.proto_class, "stage_of": self.stage_of}

    # ------------------------------------------------------------------ objectives
    def cluster_sep_cost(self, feats, targets):
        pooled, _ = self._similarities(feats)
        oh = self.proto_class[:, targets].t()
        return -(pooled * oh).max(1).values.mean(), \
            (pooled * (1 - oh)).max(1).values.mean()

    def l1_last(self):
        return (self.last.weight * (1 - self.proto_class.t())).abs().sum()

    def diversity_cost(self):
        """Zero: cross-stage prototypes are diverse by construction, and penalising
        WITHIN-stage pairs here would confound this experiment with the one in
        protopnet_diverse.py. Kept so the training loop can call it unconditionally."""
        return self.prototypes.new_zeros(())


# ----------------------------------------------------------------------- push
@torch.no_grad()
def push_multiscale(model, loader, device, max_batches=None):
    """Project each prototype onto its nearest same-class patch AT ITS OWN STAGE.

    The single-scale push in train.py assumes one feature map, so it cannot be reused:
    a stride-8 prototype must be projected onto stride-8 patches.
    """
    head = model.head
    P, D = head.P, head.proto_dim
    best_d = torch.full((P,), float("inf"))
    best_v = torch.zeros(P, D)
    ppc = head.ppc

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        feats = pyramid_forward(model.backbone, x)
        for si, s in enumerate(head.stages):
            z = head.add_ons[si](feats[s])
            B, _, H, W = z.shape
            zf = z.permute(0, 2, 3, 1).reshape(B, H * W, D)
            for c in torch.unique(y).tolist():
                sel = zf[(y == c)].reshape(-1, D)
                if sel.numel() == 0:
                    continue
                idx = [p for p in range(c * ppc, (c + 1) * ppc)
                       if int(head.stage_of[p]) == si]
                for p in idx:
                    d = ((sel - head.prototypes[p].view(1, D)) ** 2).sum(1)
                    md, mi = d.min(0)
                    if md < best_d[p]:
                        best_d[p] = md.cpu()
                        best_v[p] = sel[mi].detach().cpu()

    missed = torch.isinf(best_d)
    if (~missed).any():
        head.prototypes.data[~missed] = \
            best_v[~missed].to(head.prototypes.device).view(-1, D, 1, 1)
    per_stage = {int(s): int(((head.stage_of == si) & ~missed).sum())
                 for si, s in enumerate(head.stages)}
    return {"pushed": int((~missed).sum()), "missed": int(missed.sum()),
            "per_stage": per_stage}


# --------------------------------------------------------------------- attribution
def multiscale_grad_attr(model, x, target, stage=None, smooth=0):
    """Pixel-resolution attribution, optionally for ONE stage's prototypes only.

    stage=None -> all prototypes.  stage=i -> only the prototypes living at stages[i].

    Per-stage maps are the point of this head: the fine-stage map should be visibly
    tighter than the coarse one. If it is not, the receptive-field model is wrong.
    """
    head = model.head
    n = max(1, smooth)
    sigma = 0.10 if smooth else 0.0
    total = torch.zeros_like(x[:, :1])
    for _ in range(n):
        xi = x if sigma <= 0 else x + torch.randn_like(x) * sigma
        xi = xi.clone().detach().requires_grad_(True)
        pooled, _ = head._similarities(pyramid_forward(model.backbone, xi))
        w = head.last.weight
        if head.pip_sparsity:
            w = F.relu(w)
        wy = w[target]                                            # (B,P)
        if stage is not None:
            wy = wy * (head.stage_of == stage).float().unsqueeze(0)
        g, = torch.autograd.grad((pooled * wy).sum(), xi)
        total = total + (g * xi).abs().sum(1, keepdim=True).detach()
    return total / n
