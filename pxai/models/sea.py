"""SEA — Spatially-Exact Additive head.

The forward pass is an additive decomposition of the logit over a spatial grid:

    f_c(x)  =  a_c  +  sum_p  phi_{c,p}(x)

where `a_c` is an input-independent per-class prior and `phi_{c,p}` is produced
by a shared 1x1 readout applied at stride `r` (4, 8, 16 or 32 -- your choice,
independent of the backbone's stride 32). The attribution map IS the summands.
Nothing is estimated, sampled, integrated or regressed.

Three properties, each provable in one line and each checked by
`test_sea_axioms.py`:

  P1 COMPLETENESS   sum_p phi_{c,p} = f_c(x) - a_c, exactly, for every x and c.
                    (IG satisfies this only as m -> inf; LIME only up to the
                    surrogate's regression residual.)

  P2 SHAPLEY EXACTNESS  For the game v_c(S) = a_c + sum_{p in S} phi_{c,p} the
                    marginal contribution of p is phi_{c,p} for EVERY coalition
                    S, so Sh_p(v_c) = phi_{c,p}. The head returns exact Shapley
                    values of its own value function in one forward pass:
                    efficiency, symmetry, dummy and linearity hold by
                    construction, with no coalition sampler and no surrogate.

  P3 CONTRASTIVE EXACTNESS  f_c - f_c' = (a_c - a_c') + sum_p (phi_{c,p} -
                    phi_{c',p}), so the explanation of the DECISION is exact
                    too, not only the explanation of one logit.

WHAT THIS DOES NOT CLAIM. The decomposition is over output locations, and each
phi_p still depends on pixels outside cell p through (i) the evidence stream's
receptive field and (ii) the global context vector. So P2 is exactness with
respect to the model's internal game, NOT with respect to pixel-space masking.
Two guards are built in: `context: none` (the default) removes (ii)
entirely -- check 4 of the axiom suite measures FiLM's influence at 0.195
relative, which is not negligible -- and the adversarial probe (`context_adv > 0`) measures how much class information
survives in the context vector. Report the probe accuracy. If it is near 1/K,
every discriminative bit lives in phi.

RECEPTIVE FIELD. `max_evidence_stride` selects which backbone stages feed the
evidence stream. Note that mobilevit_* runs global self-attention inside its
stride-8/16/32 blocks, so NO mobilevit variant has a bounded receptive field.
For the locality certificate use a pure-conv backbone (efficientnet_lite0,
ghostnet_100).

SPECIAL CASE. With `readout: linear`, `context: none` and stride 32, SEA
reduces exactly to CAM. That is deliberate: it makes stride, readout capacity
and the concentration prior three independently ablatable factors over a known
baseline.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone


# --------------------------------------------------------------------------- #
# context leakage adversary
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float):
    return _GradReverse.apply(x, lam)


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
class _DWSep(nn.Module):
    """3x3 depthwise + 1x1 pointwise, GroupNorm, GELU. ~D*9 + D*D params."""

    def __init__(self, dim: int, groups: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(min(groups, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class SEAHead(nn.Module):
    """Evidence stream + optional FiLM context + additive readout."""

    def __init__(self, in_channels: Sequence[int], num_classes: int,
                 ctx_channels: int, dim: int = 64, depth: int = 2,
                 readout: str = "mlp", context: str = "none",
                 context_adv: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.readout_kind = readout
        self.context_kind = context
        self.context_adv = float(context_adv)

        self.fuse = nn.Conv2d(int(sum(in_channels)), dim, 1, bias=False)
        self.body = nn.Sequential(*[_DWSep(dim) for _ in range(depth)])

        if context == "film":
            self.film = nn.Sequential(
                nn.Linear(ctx_channels, 2 * dim), nn.GELU(),
                nn.Linear(2 * dim, 2 * dim))
        elif context == "none":
            self.film = None
        else:
            raise ValueError(f"context must be film|none, got {context!r}")

        # leakage probe: a linear classifier on the context vector, trained to
        # succeed while the gradient reversal trains the backbone to defeat it.
        self.probe = nn.Linear(ctx_channels, num_classes) if context == "film" else None

        if readout == "linear":
            self.readout = nn.Conv2d(dim, num_classes, 1, bias=False)
        elif readout == "mlp":
            self.readout = nn.Sequential(
                nn.Conv2d(dim, dim, 1), nn.GELU(),
                nn.Conv2d(dim, num_classes, 1, bias=False))
        else:
            raise ValueError(f"readout must be linear|mlp, got {readout!r}")

        # a_c: the input-independent class prior. Everything input-dependent is
        # in phi, which is what makes P1 a decomposition of the whole logit.
        self.prior = nn.Parameter(torch.zeros(num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        """Re-randomise (Quantus MPRT randomises modules it can find; this makes
        the prior and the readout randomisable too)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.prior.zero_()

    def phi(self, feats: List[torch.Tensor], ctx: torch.Tensor,
            size) -> torch.Tensor:
        """-> (B, K, h, w): the per-location class contributions."""
        ups = [f if f.shape[-2:] == size else
               F.interpolate(f, size=size, mode="bilinear", align_corners=False)
               for f in feats]
        e = self.body(self.fuse(torch.cat(ups, dim=1)))                  # (B,D,h,w)

        if self.film is not None:
            gb = self.film(ctx)                                          # (B,2D)
            gamma, beta = gb[:, :self.dim], gb[:, self.dim:]
            e = e * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        raw = self.readout(e)                                            # (B,K,h,w)
        # 1/sqrt(P) keeps sum_p phi at O(1) at init regardless of stride, so the
        # same lr works for r = 4 and r = 32. Any positive constant preserves P1.
        return raw / math.sqrt(size[0] * size[1])


class SEANet(nn.Module):
    """backbone pyramid -> SEA head. `.explain(x)['contrib_map']` is the map."""

    def __init__(self, cfg):
        super().__init__()
        s = cfg["model"].get("sea", {})
        self.stride = int(s.get("stride", 8))
        self.max_evidence_stride = int(s.get("max_evidence_stride", 16))

        # train.py:_compute_loss only receives (model, kind, x, y), so the loss
        # weights ride on the model rather than through a new argument.
        self.loss_w = dict(s.get("loss", {}))

        self.backbone = build_backbone(cfg)
        net = self.backbone.net                       # timm features_only model
        red = list(net.feature_info.reduction())
        ch = list(net.feature_info.channels())
        self.idx = [i for i, r in enumerate(red) if r <= self.max_evidence_stride]
        if not self.idx:
            raise ValueError(
                f"no backbone stage at stride <= {self.max_evidence_stride}; "
                f"available strides are {red}")
        self.ctx_idx = len(red) - 1
        self.reductions = red

        self.head = SEAHead(
            in_channels=[ch[i] for i in self.idx],
            num_classes=cfg["model"]["num_classes"],
            ctx_channels=ch[self.ctx_idx],
            dim=int(s.get("dim", 64)),
            depth=int(s.get("depth", 2)),
            readout=s.get("readout", "mlp"),
            context=s.get("context", "none"),
            context_adv=float(s.get("context_adv", 0.0)),
        )

    # -- internals ---------------------------------------------------------- #
    def _pyramid(self, x):
        return self.backbone.net(x)

    def _forward_parts(self, x):
        pyr = self._pyramid(x)
        H, W = x.shape[-2:]
        size = (max(1, H // self.stride), max(1, W // self.stride))
        ctx = F.adaptive_avg_pool2d(pyr[self.ctx_idx], 1).flatten(1)
        phi = self.head.phi([pyr[i] for i in self.idx], ctx, size)
        return phi, ctx, pyr

    # -- public API --------------------------------------------------------- #
    def forward(self, x):
        phi, _, _ = self._forward_parts(x)
        return self.head.prior.view(1, -1) + phi.sum(dim=(-2, -1))

    def forward_full(self, x):
        """Everything the loss needs, in one pass."""
        phi, ctx, pyr = self._forward_parts(x)
        logits = self.head.prior.view(1, -1) + phi.sum(dim=(-2, -1))
        probe = None
        if self.head.probe is not None and self.head.context_adv > 0:
            probe = self.head.probe(grad_reverse(ctx, self.head.context_adv))
        return {"logits": logits, "phi": phi, "ctx": ctx, "probe_logits": probe,
                "features": pyr[-1]}

    def features(self, x):
        return self.backbone(x)

    def explain(self, x):
        """(B, K, h, w) signed contribution map. One forward pass, all classes."""
        phi, _, _ = self._forward_parts(x)
        return {"contrib_map": phi, "prior": self.head.prior.detach()}


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def sea_loss(model, x, y, w: dict | None = None):
    """CE + concentration + total variation (+ adversarial context probe).

    The two attribution priors are BOX-BLIND on purpose: neither term sees a
    ground-truth annotation, so pointing-game / concentration / IoU remain
    honest held-out metrics. Do not add a box-supervised term to the headline
    model -- if you want that number, train it as a clearly-labelled separate
    arm and say so.

    concentration: with sum_p phi = f - a pinned by the data, an L1 penalty
    stops biting once phi >= 0 (since sum|phi| = |sum phi| there). The quantity
    that keeps biting is the participation ratio PR = 1 / sum_p pbar_p^2 of the
    normalised positive evidence -- the effective number of active cells. We
    hinge it at `tau` (as a fraction of the grid) so there is pressure to be at
    least this concentrated and none to collapse to a single cell.
    """
    w = w or {}
    lam_c = float(w.get("conc", 0.0))
    lam_tv = float(w.get("tv", 0.0))
    conc_on = w.get("conc_on", "abs")
    lam_x = float(w.get("cross", 0.0))
    lam_p = float(w.get("probe", 0.0))
    tau = float(w.get("tau", 0.25))
    eps = 1e-8

    out = model.forward_full(x)
    logits, phi = out["logits"], out["phi"]
    loss = F.cross_entropy(logits, y)
    stats = {"ce": float(loss.detach())}

    B, K, h, wd = phi.shape
    P = h * wd
    phi_y = phi.gather(1, y.view(-1, 1, 1, 1).expand(-1, 1, h, wd)).squeeze(1)

    # Both priors are self-scaling. The 1/(1+CE) factor is an implicit warmup:
    # while the model is still learning to classify (CE large) the priors are
    # suppressed; once CE is small -- which on this dataset it will be, accuracy
    # is saturated at 0.985-1.000 -- they become the dominant term. No epoch
    # plumbing, so train.py needs no extra argument.
    ramp = 1.0 / (1.0 + loss.detach())
    stats["ramp"] = float(ramp)

    if lam_c > 0:
        # NB: measuring concentration on the POSITIVE part only leaves a
        # loophole -- the model satisfies it by dumping compensating NEGATIVE
        # mass across the frame, which |phi|-based conc then counts. Measured
        # on |phi| the loophole is closed. See SEA_DESIGN.md section 6.
        pos = (phi_y.abs() if conc_on == "abs" else phi_y.clamp(min=0)).flatten(1)
        pbar = pos / pos.sum(1, keepdim=True).clamp_min(eps)
        hhi = (pbar * pbar).sum(1).clamp_min(1.0 / P)            # [1/P, 1]
        pr_frac = (1.0 / hhi) / P                                # [1/P, 1]
        loss = loss + lam_c * ramp * F.relu(pr_frac - tau).mean()
        stats["pr_frac"] = float(pr_frac.mean().detach())

    if lam_tv > 0:
        # normalised by mean |phi| so this is a dimensionless roughness, not a
        # magnitude penalty -- otherwise it just shrinks phi and does nothing.
        tv = ((phi_y[:, 1:, :] - phi_y[:, :-1, :]).abs().mean()
              + (phi_y[:, :, 1:] - phi_y[:, :, :-1]).abs().mean())
        tv = tv / phi_y.abs().mean().clamp_min(eps)
        loss = loss + lam_tv * ramp * tv
        stats["tv_rel"] = float(tv.detach())

    if lam_x > 0:
        mask = torch.ones_like(phi, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1, 1, 1).expand(-1, 1, h, wd), False)
        loss = loss + lam_x * phi[mask].clamp(min=0).mean()

    if lam_p > 0 and out["probe_logits"] is not None:
        # GRL already flipped the sign for the encoder; the probe itself
        # minimises this, so it learns to read the context as well as it can.
        pl = F.cross_entropy(out["probe_logits"], y)
        loss = loss + lam_p * pl
        stats["probe_ce"] = float(pl.detach())

    return loss, stats
