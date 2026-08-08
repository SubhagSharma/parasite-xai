"""Family-shared attention head — one attention map per morphological FAMILY, with a
softmax over that family's values.

===============================================================================
WHY: WHAT THE CURRENT HEAD DOES WRONG
===============================================================================
ConceptPartHead gives each of the 23 concepts its own attention slot, and each concept
is a separate binary prediction. Measured outcome: above the resolution threshold the
slots become cross-species CONSISTENT (2/12 -> 12/12) while remaining anatomically
WRONG. Visual confirmation on the DINOv2 arm:

    operculum (a polar lid)        -> attends to the egg INTERIOR
    contents=unembryonated         -> attends to the shell RIM and to debris outside
                                      the annotation box

The mechanism is simple and it is not a bug in the code. Concept k is predicted from
slot k's pooled feature, and the loss only requires c_k to be CORRECT. Any region whose
features predict c_k is equally acceptable.

`contents=unembryonated` is true for 5 of 11 species, so the slot must output 1 for
those 5 and 0 for the other 6. Ascaris's shell rim separates those groups exactly as
well as its interior does. Nothing prefers the interior. Twenty-three independent binary
tests, each satisfiable from anywhere in the image.

===============================================================================
FIX 1: FAMILY-SHARED ATTENTION  (architectural, dataset-agnostic)
===============================================================================
The 23 concepts are not independent. They are 9 FAMILIES, and within a family the values
are mutually exclusive:

    shell_texture    in {mammillated, radially_striated, rough, smooth, striated}
    size_band        in {very_small, small, medium, large, very_large}
    contents         in {cleaved_embryo, larva, miracidium, oncosphere, unembryonated}
    shell_thickness  in {thin, thick}
    symmetry         in {symmetric, asymmetric}
    operculum, has_polar_plugs, has_polar_filaments, has_polar_knob   -- binary singletons

Give each FAMILY one attention map, and predict the family's value with a SOFTMAX over
its members. The pooled feature must then carry enough information to say WHICH texture,
not merely "is it smooth, yes or no".

Why that helps: five-way discrimination from a single region is much harder to satisfy
from an incidental correlate than five independent binaries are. And it matches anatomy
for free -- shell texture is a property OF THE SHELL, so one region with five possible
answers points where the texture actually differs.

This covers 18 of 23 concepts and makes no claim about parasites. It is a claim about the
structure of categorical concept sets, and would apply unchanged to CUB attributes or any
dataset with mutually exclusive concept groups. That generality is the point.

**Honest limit: this SHRINKS the space of available cheats, it does not close it.** A
region that correlates with texture across species could still work. Five-way is harder
to fake than five binaries; it is not a guarantee.

===============================================================================
FIX 2: GEOMETRIC PRIORS  (domain instantiation, optional)
===============================================================================
Family-sharing does nothing for the four binary singletons -- and those are precisely the
features of interest: operculum, polar plugs, polar filaments, polar knob.

For those, CDC DPDx already states WHERE: polar means at an extremity of the egg's major
axis. `probe_anatomy` used that to EVALUATE. This uses it to SUPERVISE: penalise
attention mass that is not near an extremity.

    L_polar = mean over polar families of  E_A[ 1 - r ]

where r is the attention-weighted distance from the image centre, normalised so r=1 at
the corner. The term is 0 when all mass sits at the periphery and 1 when it sits at the
centre. Radius rather than a named axis, so the prior is ROTATION-INVARIANT -- eggs are
arbitrarily oriented -- and needs no annotation box during training. On crops and both
ROI variants the extraction centres the egg, so image centre is a good proxy for egg
centre; on whole images it is not, and the prior should not be used there.

This encodes a published definition, not a per-image label -- it is the same knowledge a
parasitologist brings. The line to hold: **priors that encode a published definition are
legitimate; priors tuned until the maps look right are not.** If the penalty ends up being
adjusted because an operculum map is 20px off, that is fitting the answer, and the result
should not be reported.

Default w_polar = 0, so an unconfigured model reproduces family-sharing alone and the two
fixes can be ablated separately.

===============================================================================
WHAT NEITHER FIX GUARANTEES
===============================================================================
Only spatial supervision eliminates the cheat. If both fixes fail, the honest conclusion
is that class-level concept supervision cannot specify WHERE, and ~100 annotated part
boxes (10 per species, held-out evaluation) is the only remaining route. Knowing that
these two failed is what would justify that annotation effort.

    model:
      kind: family_parts
      family_parts:
        concepts_csv: ../Data/Chula-ParasiteEgg-11/concepts_v3.csv
        slot_dim: 128
        w_compact: 0.1
        w_polar: 0.0        # fix 2; 0 disables
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# families whose members lie at an extremity of the egg's major axis, per CDC DPDx
POLAR_FAMILIES = ("operculum", "has_polar_plugs", "has_polar_filaments",
                  "has_polar_knob")


def parse_families(path: str):
    """-> (family_names, members) where members[f] is the list of concept indices in
    family f, in the same order load_concept_table produces.

    Reconstructs the one-hot expansion the loader performs (9 raw CSV columns -> 23
    concepts) so the indices line up with the concept table.
    """
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0][1:], [r[1:] for r in rows[1:]]
    fam_names, members, k = [], [], 0
    for j, col in enumerate(header):
        vals = sorted({r[j] for r in data})
        if set(vals) <= {"0", "1"}:                 # already binary
            fam_names.append(col)
            members.append([k])
            k += 1
        else:                                       # one-hot expanded
            fam_names.append(col)
            members.append(list(range(k, k + len(vals))))
            k += len(vals)
    return fam_names, members, k


class FamilyPartHead(nn.Module):
    """One attention map per family; a softmax over that family's mutually exclusive
    values. Binary singleton families keep a sigmoid."""

    def __init__(self, in_channels: int, num_classes: int,
                 family_names: Sequence[str], members: Sequence[Sequence[int]],
                 num_concepts: int, slot_dim: int = 128,
                 w_compact: float = 0.1, w_polar: float = 0.0,
                 hard_bottleneck: bool = True):
        super().__init__()
        self.fam_names = list(family_names)
        self.members = [list(m) for m in members]
        self.F = len(self.members)
        self.K = num_concepts
        self.num_classes = num_classes
        self.hard = hard_bottleneck
        self.w_compact = w_compact
        self.w_polar = w_polar

        self.attn = nn.Conv2d(in_channels, self.F, 1)      # ONE map per FAMILY
        self.proj = nn.Conv2d(in_channels, slot_dim, 1)
        # each family reads only its own slot; the head over that slot is |family|-way
        self.fam_w = nn.ParameterList(
            [nn.Parameter(torch.randn(len(m), slot_dim) * 0.02) for m in self.members])
        self.fam_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(m))) for m in self.members])
        self.classifier = nn.Linear(num_concepts, num_classes)

        pol = torch.zeros(self.F)
        for i, n in enumerate(self.fam_names):
            if n in POLAR_FAMILIES:
                pol[i] = 1.0
        self.register_buffer("is_polar", pol)

    # ------------------------------------------------------------------ forward
    def _slots(self, feat):
        B, _, H, W = feat.shape
        A = F.softmax(self.attn(feat).reshape(B, self.F, H * W), dim=-1)
        A = A.reshape(B, self.F, H, W)
        z = self.proj(feat)
        v = torch.einsum("bfhw,bdhw->bfd", A, z)           # (B,F,slot_dim)
        return A, v

    def forward(self, feat, concept_intervene=None):
        A, v = self._slots(feat)
        B = feat.shape[0]
        c_logit = feat.new_zeros(B, self.K)
        c = feat.new_zeros(B, self.K)
        for i, m in enumerate(self.members):
            lg = v[:, i] @ self.fam_w[i].t() + self.fam_b[i]      # (B,|family|)
            c_logit[:, m] = lg
            # softmax when the family's values are mutually exclusive; sigmoid for
            # binary singletons
            c[:, m] = F.softmax(lg, dim=1) if len(m) > 1 else torch.sigmoid(lg)
        if concept_intervene is not None:
            msk = ~torch.isnan(concept_intervene)
            c = torch.where(msk, concept_intervene.nan_to_num(), c)
        return self.classifier(c if self.hard else c_logit), c_logit

    @torch.no_grad()
    def explain(self, feat):
        A, v = self._slots(feat)
        return {"attn_family": A, "families": self.fam_names, "members": self.members}

    def concept_map(self, feat, k: int, size=None):
        """Attention for concept k -- i.e. for the FAMILY that owns k.

        Concepts within a family share one map by construction. That is the point: shell
        texture is a property of the shell, so `shell_texture=smooth` and `=striated`
        should be read from the same place.
        """
        fi = next(i for i, m in enumerate(self.members) if k in m)
        with torch.no_grad():
            A, _ = self._slots(feat)
        m = A[:, fi:fi + 1]
        return F.interpolate(m, size=size, mode="bilinear",
                             align_corners=False) if size else m

    # ------------------------------------------------------------- regularisers
    def part_loss(self, feat):
        """compact + optional polar prior. Needs no annotation box: the polar term
        is radius-based and therefore rotation-invariant."""
        if self.w_compact <= 0 and self.w_polar <= 0:
            return feat.new_zeros(())
        A, _ = self._slots(feat)
        B, Fn, H, W = A.shape
        dev = A.device
        ys = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, Fn, H, W)
        xs = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, Fn, H, W)

        total = feat.new_zeros(())
        if self.w_compact > 0:
            cy = (A * ys).sum((-2, -1))
            cx = (A * xs).sum((-2, -1))
            var = (A * ((ys - cy[..., None, None]) ** 2
                        + (xs - cx[..., None, None]) ** 2)).sum((-2, -1))
            total = total + self.w_compact * var.mean()

        if self.w_polar > 0 and self.is_polar.sum() > 0:
            # "Polar" means at an extremity, i.e. FAR FROM THE CENTRE. Using radius
            # rather than a named axis makes the prior rotation-invariant, which matters
            # because eggs are arbitrarily oriented in the frame, and it needs no
            # annotation box at training time.
            #
            # The centre is taken as the image centre. On crops and both ROI variants
            # the extraction is centred on the egg, so image centre ~ egg centre; on
            # whole images it is not, and the prior should not be used there.
            idx = self.is_polar.nonzero(as_tuple=True)[0]
            r = torch.sqrt((ys[:, idx] - 0.5) ** 2 + (xs[:, idx] - 0.5) ** 2) / 0.7071
            # 0 when all mass sits at the periphery, 1 when it sits at the centre
            pol = (A[:, idx] * (1.0 - r.clamp(0, 1))).sum((-2, -1))
            total = total + self.w_polar * pol.mean()
        return total


def build_family_head(in_channels, num_classes, cfg_p):
    csv_path = cfg_p["concepts_csv"]
    names, members, K = parse_families(csv_path)
    n_pol = sum(1 for n in names if n in POLAR_FAMILIES)
    print(f"[family] {K} concepts in {len(members)} families "
          f"({n_pol} polar), sizes {[len(m) for m in members]}", flush=True)
    return FamilyPartHead(in_channels, num_classes, names, members, K,
                          cfg_p.get("slot_dim", 128),
                          cfg_p.get("w_compact", 0.1),
                          cfg_p.get("w_polar", 0.0),
                          cfg_p.get("bottleneck", True))
