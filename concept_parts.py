"""Spatially-grounded concept head — one attention map per DPDx morphology concept.

===============================================================================
THE IDEA
===============================================================================
The stated goal of this project is a model that points at the *operculum*, the *polar
plugs*, the *striated shell* — the features CDC DPDx actually names. Four attempts to
get there through prototypes have failed, and the measurements say why:

    prototype winner counts, trained baseline (per class, out of 5 prototypes)
      Ascaris     [0, 0, 0, 42, 0]     one prototype takes 42/42 samples
      Paragonimus [0, 0, 33, 0, 0]     33/33
      Trichuris   [0, 0, 0, 48, 0]     48/48
      mean usage balance 10%; prototype effective rank 1.03 of 5

Prototypes have to DISCOVER parts unaided. Four of five end up dead. Rank collapses.
Every fix aimed at the head (orthogonality, sparsity, Hungarian push) or the backbone
(DINOv2, multi-scale) failed to move it.

**This head does not ask the model to discover anything. It tells each slot what to
find.**

===============================================================================
WHY CLASS-LEVEL CONCEPTS ARE ENOUGH
===============================================================================
The obvious objection: `concepts_v3.csv` is class-level. `has_polar_plugs = 1` for
Trichuris says the IMAGE contains plugs, not WHERE they are. How can that supervise a
spatial attention map?

**By contrast across classes.** Trichuris and Capillaria have polar plugs; Ascaris and
Taenia do not. If slot k's pooled feature is the ONLY input to the `has_polar_plugs`
prediction, then slot k must produce evidence on plug-having species and withhold it on
the others.

The only way to satisfy that with a spatial attention map is to attend to the plug.
No per-image annotation is required — the between-class discrimination supplies it.

This is the same principle behind class-activation mapping and behind PDiscoNet's
discriminability prior, applied per-concept rather than per-class.

===============================================================================
ARCHITECTURE
===============================================================================
    A_k     = softmax_spatial( conv_k(features) )          one attention map per concept
    v_k     = sum_spatial( A_k * features )                slot k's pooled feature
    c_k     = sigmoid( w_k . v_k )                         concept k from slot k ALONE
    logit   = W . c                                        class from concepts ONLY

The critical constraint is `w_k . v_k`: concept k is predicted from a **scalar projection
of its own slot**, never from the global feature and never from another slot's. That is
what forces A_k to localise. A shared trunk would let the model predict every concept
from a global average and leave the attention maps meaningless — which is exactly what
the current CBM does (`self.pool(feat).flatten(1)`), and why it has no spatial
explanation at all.

**The attention map IS the explanation.** `A_operculum` shows the operculum. No gradient
trick, no upsampled prototype grid, no post-hoc method.

===============================================================================
REGULARISERS, AND WHY EACH IS THERE
===============================================================================
Attention maps left unconstrained collapse to a single pixel or spread to the whole
frame. Three priors, all from the part-discovery literature (van der Klis et al.,
PDiscoNet, ICCV 2023):

  compactness   penalise the spatial variance of each A_k about its own centroid.
                A morphological feature is a connected region, not scattered pixels.

  distinctness  penalise overlap between concept slots that BELONG TO THE SAME
                MORPHOLOGICAL FAMILY. `shell_texture=smooth` and `shell_texture=striated`
                describe the same anatomy and should attend to the same place; but
                `operculum` and `size_band=large` should not. Families are inferred from
                the concept names in the CSV (text before `=`), so the prior encodes
                real anatomy rather than a blanket "all slots must differ".

  presence      each concept slot should be active on at least some images per batch,
                which prevents the dead-slot failure that killed the prototypes.

===============================================================================
WHAT TO EXPECT, AND WHICH CONCEPTS WILL WORK
===============================================================================
A concept can only be localised if it discriminates. Positive rate across the 11
classes, from concepts_v3.csv:

    contents=unembryonated       ~5/11   BEST -- maximum contrast
    operculum                    ~3/11   strong
    has_polar_plugs              ~2/11   strong
    shell_texture=mammillated     1/11   strong but few positives
    symmetry=symmetric           10/11   WORST -- almost no contrast

`symmetry=symmetric` is expected to fail, and the existing CBM already shows it failing
(balanced accuracy 0.796, TPR 0.600, the only concept below 0.99). That is a prediction
this design makes which the current data already confirms.

===============================================================================
EVALUATION
===============================================================================
Unlike prototypes, these maps have a name attached, so they can be checked:

  1. does A_operculum sit inside the annotation box?  -> conc_pos, as before
  2. does A_operculum sit on the SAME PART across images of a species? -> centroid
     variance across images
  3. does a parasitologist agree it is the operculum? -> the only test that finally
     matters, and the one CUB-based part discovery cannot offer

    config:
      model:
        kind: concept_parts
        concept_parts:
          num_concepts: 23
          concepts_csv: ../Data/Chula-ParasiteEgg-11/concepts_v3.csv
          w_compact: 0.1
          w_distinct: 0.1
          w_presence: 0.1
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptPartHead(nn.Module):
    """One spatial attention slot per concept; class predicted from concepts only."""

    def __init__(self, in_channels: int, num_classes: int, num_concepts: int = 23,
                 slot_dim: int = 128, families: Optional[List[int]] = None,
                 w_compact: float = 0.1, w_distinct: float = 0.1,
                 w_presence: float = 0.1, hard_bottleneck: bool = True):
        super().__init__()
        self.K = num_concepts
        self.num_classes = num_classes
        self.hard = hard_bottleneck
        self.w_compact = w_compact
        self.w_distinct = w_distinct
        self.w_presence = w_presence

        # concepts sharing a morphological family should be allowed to co-locate;
        # concepts from different families should not. -1 = its own family.
        if families is None:
            families = list(range(num_concepts))
        self.register_buffer("families", torch.tensor(families, dtype=torch.long))

        self.proj = nn.Conv2d(in_channels, slot_dim, 1)
        self.attn = nn.Conv2d(in_channels, num_concepts, 1)      # one map per concept
        # concept k reads ONLY slot k: a per-slot scalar projection, not a shared
        # linear over concatenated slots. This is what forces A_k to localise.
        self.slot_w = nn.Parameter(torch.randn(num_concepts, slot_dim) * 0.02)
        self.slot_b = nn.Parameter(torch.zeros(num_concepts))
        self.classifier = nn.Linear(num_concepts, num_classes)

        self._last = {}

    # ------------------------------------------------------------------ forward
    def _slots(self, feat):
        B, _, H, W = feat.shape
        a = self.attn(feat).reshape(B, self.K, H * W)
        A = F.softmax(a, dim=-1).reshape(B, self.K, H, W)         # spatial attention
        z = self.proj(feat)                                       # (B, slot_dim, H, W)
        # v[b,k,:] = sum over space of A[b,k] * z[b,:]
        v = torch.einsum("bkhw,bdhw->bkd", A, z)
        return A, v

    def forward(self, feat, concept_intervene=None):
        A, v = self._slots(feat)
        c_logit = (v * self.slot_w.unsqueeze(0)).sum(-1) + self.slot_b   # (B,K)
        c = torch.sigmoid(c_logit)
        if concept_intervene is not None:
            m = ~torch.isnan(concept_intervene)
            c = torch.where(m, concept_intervene.nan_to_num(), c)
        logit = self.classifier(c if self.hard else c_logit)
        self._last = {"A": A.detach(), "c_logit": c_logit.detach()}
        return logit, c_logit

    @torch.no_grad()
    def explain(self, feat):
        """Attention maps ARE the explanation -- no gradient trick, no upsampled grid."""
        A, v = self._slots(feat)
        c_logit = (v * self.slot_w.unsqueeze(0)).sum(-1) + self.slot_b
        return {"attn": A, "c_logit": c_logit, "concepts": torch.sigmoid(c_logit)}

    def concept_map(self, feat, k: int, size=None):
        """Attention for concept k, upsampled to `size`. The deliverable: a named,
        localised morphology map."""
        with torch.no_grad():
            A, _ = self._slots(feat)
        m = A[:, k:k + 1]
        return F.interpolate(m, size=size, mode="bilinear",
                             align_corners=False) if size else m

    # ------------------------------------------------------------- regularisers
    def part_costs(self, feat):
        """-> (compact, distinct, presence). Zero when the weights are zero."""
        if max(self.w_compact, self.w_distinct, self.w_presence) <= 0:
            z = feat.new_zeros(())
            return z, z, z
        A, _ = self._slots(feat)
        B, K, H, W = A.shape
        dev = A.device

        ys = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, K, H, W)
        xs = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, K, H, W)
        cy = (A * ys).sum((-2, -1))
        cx = (A * xs).sum((-2, -1))
        # spatial variance about the slot's own centroid: a morphological feature is a
        # connected region, not scattered pixels
        var = (A * ((ys - cy[..., None, None]) ** 2
                    + (xs - cx[..., None, None]) ** 2)).sum((-2, -1))
        compact = var.mean()

        Af = A.reshape(B, K, H * W)
        ov = torch.bmm(Af, Af.transpose(1, 2))                   # (B,K,K) overlap
        same = (self.families.view(-1, 1) == self.families.view(1, -1))
        mask = (~same).float().to(dev) * (1 - torch.eye(K, device=dev))
        # only penalise overlap ACROSS morphological families: shell_texture=smooth and
        # =striated describe the same anatomy and should co-locate
        distinct = (ov * mask).sum() / (B * mask.sum().clamp_min(1))

        # every slot active somewhere in the batch -- prevents the dead-slot failure
        # that left 4 of 5 prototypes unused
        presence = 1.0 - Af.max(-1).values.max(0).values.mean()

        return compact, distinct, presence

    def part_loss(self, feat):
        c, d, p = self.part_costs(feat)
        return self.w_compact * c + self.w_distinct * d + self.w_presence * p


def families_from_csv(path: str, names: Optional[List[str]] = None) -> List[int]:
    """Group concepts by the text before '=' so that `shell_texture=smooth` and
    `shell_texture=striated` share a family and are allowed to attend to the same place.
    Concepts without '=' (operculum, has_polar_plugs) each get their own family."""
    import csv
    with open(path) as f:
        header = next(csv.reader(f))
    cols = names or [h for h in header[1:]]
    fam, seen = [], {}
    for i, c in enumerate(cols):
        key = c.split("=")[0] if "=" in c else f"__{c}"
        if key not in seen:
            seen[key] = len(seen)
        fam.append(seen[key])
    return fam
