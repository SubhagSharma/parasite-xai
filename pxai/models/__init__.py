"""Model factory — assembles backbone + interpretable head (A2 ablation)."""
from __future__ import annotations
import os

import torch.nn as nn

from ..backbones import build_backbone
from .protopnet import ProtoHead
from .protopnet_multiscale import (MultiScaleProtoHead, pyramid_forward,
                                   push_multiscale)
from .protopnet_diverse import DiverseProtoHead
from .cbm import CBMHead
from .concept_parts import ConceptPartHead, families_from_csv
from .family_parts import build_family_head
from .bcos import BcosHead
from .blackbox import BlackBox
from .sea import SEANet


class InterpretableModel(nn.Module):
    """backbone -> interpretable head. `.explain(x)` exposes the head's evidence."""

    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg["model"]["kind"]
        nc = cfg["model"]["num_classes"]
        if cfg["backbone"]["name"].startswith("dino"):
            # frozen self-supervised backbone: P2 tests whether prototype diversity
            # rises when the representation retains within-class rank
            from .dino_backbone import DinoBackbone
            self.backbone = DinoBackbone(
                grid_out=cfg["backbone"].get("grid_out", None))
        elif cfg["backbone"]["name"].startswith("bcos_"):
            # a real B-cos network: every conv is a B-cos transform, so the whole
            # net collapses to an exact input-dependent linear map. Its native
            # explanation IS grad x input, which predicts gradattr gain ~ 1.0x.
            from .bcos_backbone import BcosBackbone
            self.backbone = BcosBackbone(cfg["backbone"]["name"][5:])
        else:
            self.backbone = build_backbone(cfg)
        ch = self.backbone.out_channels

        if self.kind == "protopnet_ms":
            p = cfg["model"].get("protopnet_ms", {})
            try:
                ch_all = self.backbone.net.feature_info.channels()
            except AttributeError as e:
                raise RuntimeError(
                    "protopnet_ms needs a timm features_only backbone exposing "
                    "feature_info.channels()") from e
            stages = list(p.get("stages", [1, 3]))
            bad = [s for s in stages if s >= len(ch_all)]
            if bad:
                raise ValueError(
                    f"stage index {bad} out of range: the backbone has "
                    f"{len(ch_all)} stages with channels {ch_all}")
            self.head = MultiScaleProtoHead(
                ch_all, nc, stages,
                p.get("protos_per_class_per_stage", [2, 3]),
                p.get("proto_dim", 128), p.get("pip_sparsity", True))
            print(f"[ms] stages {stages} channels "
                  f"{[ch_all[s] for s in stages]} "
                  f"protos/class {p.get('protos_per_class_per_stage', [2, 3])}",
                  flush=True)
        elif self.kind == "protopnet_diverse":
            p = cfg["model"].get("protopnet_diverse", cfg["model"]["protopnet"])
            self.head = DiverseProtoHead(
                ch, nc,
                p.get("num_prototypes_per_class", 5),
                p.get("proto_dim", 128),
                p.get("pip_sparsity", True),
                w_orth=p.get("w_orth", 0.0),
                w_sparse=p.get("w_sparse", 0.0),
                focal=p.get("focal", False),
                diverse_push=p.get("diverse_push", False),
                w_usage=p.get("w_usage", 0.0),
                usage_tau=p.get("usage_tau", 0.5))
        elif self.kind == "protopnet":
            p = cfg["model"]["protopnet"]
            self.head = ProtoHead(ch, nc, p["num_prototypes_per_class"],
                                  p["proto_dim"], p["pip_sparsity"])
        elif self.kind == "family_parts":
            p = cfg["model"].get("family_parts",
                                 cfg["model"].get("concept_parts", {}))
            self.head = build_family_head(ch, nc, p)
        elif self.kind == "concept_parts":
            p = cfg["model"].get("concept_parts", cfg["model"].get("cbm", {}))
            cp = p.get("concepts_csv")
            # load_concept_table one-hot expands the 9 raw CSV columns into 23
            # concepts, so families must come from the EXPANDED names -- reading the
            # raw header gives 9 families for 23 slots and the distinctness prior then
            # masks arbitrary pairs.
            fam = families_from_csv(cp) if cp and os.path.exists(cp) else None
            if fam:
                print(f"[parts] {len(fam)} concepts, "
                      f"{len(set(fam))} morphological families", flush=True)
            self.head = ConceptPartHead(
                ch, nc, p.get("num_concepts", 23), p.get("slot_dim", 128), fam,
                p.get("w_compact", 0.1), p.get("w_distinct", 0.1),
                p.get("w_presence", 0.1), p.get("bottleneck", True))
        elif self.kind == "cbm":
            c = cfg["model"]["cbm"]
            self.head = CBMHead(ch, nc, c["num_concepts"], c["bottleneck"])
        elif self.kind == "bcos":
            self.head = BcosHead(ch, nc)
        else:
            raise ValueError(f"unknown interpretable kind {self.kind!r}")

    def forward(self, x):
        # protopnet_ms attaches prototypes at several backbone depths, so its head
        # takes the whole feature pyramid. Every other head takes the last map, and
        # `features()` is left alone so no probe or figure has to change.
        if self.kind == "protopnet_ms":
            out = self.head(pyramid_forward(self.backbone, x))
        else:
            feat = self.backbone(x)
            out = self.head(feat)
        return out[0] if isinstance(out, tuple) else out

    def features(self, x):
        return self.backbone(x)

    def explain(self, x):
        return self.head.explain(self.backbone(x))


def build_model(cfg):
    if cfg["model"]["kind"] == "sea":
        return SEANet(cfg)
    if cfg["model"]["kind"] == "blackbox":
        return BlackBox(cfg["model"]["num_classes"],
                        cfg["backbone"]["name"], cfg["backbone"]["pretrained"])
    return InterpretableModel(cfg)
