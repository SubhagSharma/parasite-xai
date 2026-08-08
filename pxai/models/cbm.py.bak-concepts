"""Concept Bottleneck Model head (Koh et al. 2020).

Predicts human-readable morphology concepts first (operculum, shell texture,
size class, polar knob, ...), then the diagnosis from concepts ONLY. The hard
bottleneck makes the label provably a function of inspectable concepts, and
enables test-time intervention: a clinician edits a concept and the prediction
updates.

`concept_source`:
  - "labels"   : supervised concepts (needs concept annotations per image)
  - "labelfree": Label-free CBM (Oikarinen et al. 2023) — concepts mined from an
                 LLM and grounded with CLIP; supply `concept_bank` projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBMHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_concepts: int = 16,
                 hard_bottleneck: bool = True):
        super().__init__()
        self.num_concepts = num_concepts
        self.hard = hard_bottleneck
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.concept = nn.Linear(in_channels, num_concepts)          # x -> concept logits
        # label depends ONLY on concepts -> faithful, inspectable, intervenable
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, feat, concept_intervene: torch.Tensor | None = None):
        v = self.pool(feat).flatten(1)
        c_logit = self.concept(v)
        c = torch.sigmoid(c_logit)
        if concept_intervene is not None:
            # clinician overrides: NaN entries keep model value, others are forced
            mask = ~torch.isnan(concept_intervene)
            c = torch.where(mask, concept_intervene.nan_to_num(), c)
        feats_for_label = c if self.hard else c_logit
        logit = self.classifier(feats_for_label)
        return logit, c_logit

    @torch.no_grad()
    def explain(self, feat):
        _, c_logit = self.forward(feat)
        contrib = self.classifier.weight.unsqueeze(0) * torch.sigmoid(c_logit).unsqueeze(1)
        # contrib[b, class, concept] = how much each concept pushes each class
        return {"concepts": torch.sigmoid(c_logit), "class_concept_contrib": contrib,
                "concept_weight": self.classifier.weight}


def concept_loss(c_logit, c_target, lam: float = 0.5):
    """BCE concept supervision (only when concept labels exist)."""
    if c_target is None:
        return c_logit.new_zeros(())
    return lam * F.binary_cross_entropy_with_logits(c_logit, c_target.float())
