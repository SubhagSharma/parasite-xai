"""
run_p2_dino_prototypes.py — the interventional test of the rank bound.

WHAT P1 SHOWED, AND WHY IT IS NOT ENOUGH
----------------------------------------
    supervised MobileViT stage 4   within-class rank  3.16 of 384
    DINOv2 self-supervised                            11.40 of 384
    supervised MobileViT stage 1                      20.78 of  48

DINOv2 preserves 3.6x more within-class rank than a supervised backbone of the same
width. That is consistent with the hypothesis -- but it is a CORRELATION between two
properties of backbones. It does not show that rank is what bounds prototype diversity.

THE INTERVENTIONAL TEST
-----------------------
Train the SAME ProtoPNet head on a frozen DINOv2 backbone. Nothing else changes: same
prototypes per class, same losses, same push, no diversity penalty of any kind.

    measured on the supervised backbone:  prototype effective rank  1.02 of 5
                                          available embedding rank  1.34

    prediction: on DINOv2, prototype effective rank rises WITHOUT any diversity term,
    because there is more for the prototypes to differ on.

WHY FROZEN
If the backbone is fine-tuned, the classification objective will collapse its rank
during training and the experiment measures nothing. Freezing keeps the representation
at the rank P1 measured. It also makes the run cheap -- only the head trains.

WHAT EACH OUTCOME MEANS
    prototype rank rises  -> the bound is CAUSAL, not correlational. The diversity
                             failure is a backbone problem, every head-level fix
                             (orthogonality, sparsity, diverse push) was aimed at the
                             wrong layer, and Objective 2 is justified.
    prototype rank flat   -> rank is NOT what bounds prototype diversity. The hypothesis
                             is refuted and the research direction should be abandoned
                             rather than defended. Report it.

A CONFOUND TO WATCH
DINOv2 has a patch size of 14, so a 224px input gives a 16x16 grid rather than 7x7 --
2.6x more spatial positions. If prototype rank rises, part of that could be having more
distinct patches to choose from rather than richer ones. The control is `--grid-match`,
which average-pools the DINOv2 feature map to 7x7 first, matching the supervised grid
exactly. Run both; if the effect survives grid matching, it is about rank.

    python run_p2_dino_prototypes.py --write-configs
    # then
    python -u -m pxai.train --config configs/generated/dino_protopnet_120ep.yaml
    python -u probe_prototype_diversity.py --device cuda --runs dino_protopnet_120ep
"""
from __future__ import annotations

import argparse
import copy
import os

BACKBONE_SRC = '''"""Frozen DINOv2 backbone exposing pxai's interface. NEW FILE."""
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
'''

REGISTER_OLD = """        if cfg["backbone"]["name"].startswith("bcos_"):"""
REGISTER_NEW = """        if cfg["backbone"]["name"].startswith("dino"):
            # frozen self-supervised backbone: P2 tests whether prototype diversity
            # rises when the representation retains within-class rank
            from .dino_backbone import DinoBackbone
            self.backbone = DinoBackbone(
                grid_out=cfg["backbone"].get("grid_out", None))
        elif cfg["backbone"]["name"].startswith("bcos_"):"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="roi477_protopnet_120ep")
    ap.add_argument("--write-configs", action="store_true")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bb = os.path.join(here, "pxai", "models", "dino_backbone.py")
    if os.path.exists(bb):
        print("  pxai/models/dino_backbone.py exists, left alone")
    else:
        os.makedirs(os.path.dirname(bb), exist_ok=True)
        open(bb, "w").write(BACKBONE_SRC)
        print("  wrote pxai/models/dino_backbone.py")

    init = os.path.join(here, "pxai", "models", "__init__.py")
    s = open(init).read()
    if "DinoBackbone" in s:
        print("  models/__init__.py already registers DinoBackbone")
    elif REGISTER_OLD in s:
        open(init, "w").write(s.replace(REGISTER_OLD, REGISTER_NEW, 1))
        print("  registered DinoBackbone in models/__init__.py")
    else:
        print("  ** could not register: expected the bcos_ branch. Add by hand:")
        print(REGISTER_NEW)

    if not a.write_configs:
        print("\n--write-configs not given; no configs written.")
        return

    import yaml
    base = yaml.safe_load(open(os.path.join(here, "configs", "generated",
                                            f"{a.src}.yaml")))
    for name, grid in (("dino_protopnet_120ep", None),
                       ("dino_protopnet_g7_120ep", 7)):
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "dinov2_vits14"
        c["backbone"]["pretrained"] = True
        if grid:
            c["backbone"]["grid_out"] = grid
        c["output_dir"] = f"./runs/{name}"
        with open(os.path.join(here, "configs", "generated", f"{name}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print(f"  {name:<28} grid_out={grid or 'native (16x16)'}")

    print("""
NEXT
  python -c "import pxai.models; print('imports OK')"
  python -u preflight_learns.py --config configs/generated/dino_protopnet_120ep.yaml --device cuda

  Then train BOTH arms (~1.5h each; only the head trains, the backbone is frozen):
    dino_protopnet_120ep     native 16x16 grid
    dino_protopnet_g7_120ep  pooled to 7x7 -- the GRID CONTROL

  Then the number that decides it:
    python -u probe_prototype_diversity.py --device cuda \\
        --runs dino_protopnet_120ep,dino_protopnet_g7_120ep,roi477_protopnet_120ep

  BASELINE: roi477_protopnet_120ep has prototype effective rank 1.02 of 5.
    rises on BOTH dino arms  -> the rank bound is causal. Objective 2 is justified and
                                every head-level diversity fix was aimed at the wrong layer.
    rises only on the 16x16 arm -> it was grid size, not rank. Weaker, and honest.
    flat on both             -> hypothesis REFUTED. Report it and change direction.""")


if __name__ == "__main__":
    main()
