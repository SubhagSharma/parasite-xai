"""B-cos backbone wrapper — a real B-cos network behind pxai's existing interface.

WHY THIS IS WORTH BUILDING
==========================
Part I evaluates a **B-cos-style linear head** on a standard MobileViT-XS backbone. That
is not a B-cos network. Böhle et al. (2022) replace *every* convolution with a B-cos
transform and remove all biases and normalisation, so the whole network collapses to an
exact input-dependent linear map:

    logit_y(x) = W_y(x) . x        exactly, not approximately

**That claim makes a sharp prediction about this project's central result.**

If the network really is exactly linear in x for a given input, then

    d logit_y / dx  .  x  =  W_y(x) . x  =  the native explanation

So gradient x input is *identically* the native explanation, and the Part II gradient
read-out should give **exactly zero gain**. On the B-cos-style head it gives 2.2x.

    gain ~ 1.0  -> the head-only variant was not a real B-cos, AND the Part II
                   framework is validated by a case where it correctly predicts NO
                   improvement. That is the strongest available control: a method that
                   only ever improves things is not measuring anything.
    gain > 1    -> even a real B-cos network has a read-out gap, contradicting its own
                   design claim. A larger finding than anything in Part II.

Either outcome is publishable. Neither is obtainable from the head-only variant.

THE 3-CHANNEL DECISION
======================
Canonical B-cos takes a 6-channel [x, 1-x] encoding so negative evidence has somewhere
to live. pxai's data pipeline, every probe, `batch_visualise`, `probe_gradattr`,
`probe_prototype_diversity` and the denormalisation in every figure all assume 3
channels.

Rather than change all of that, this wrapper **accepts 3-channel input and builds the
6-channel encoding internally**. Consequences, stated so they can be reported:

  * every existing probe works unmodified
  * the attribution returned is with respect to the 3-channel input, obtained by
    summing the 6-channel gradient over the [x, 1-x] pair -- which is the correct
    chain rule, since d(1-x)/dx = -1
  * it is a wrapper detail, not a change to the B-cos property: the network's internal
    computation is unmodified and still exactly linear in its own 6-channel input

INPUT NORMALISATION
===================
B-cos expects input in [0,1], NOT ImageNet-normalised. The wrapper un-normalises using
the ImageNet statistics pxai applies, so `data.py` needs no change. If you later train
with a different normalisation, update MEAN/STD here.

INSTALL
    pip install bcos

    python bcos_backbone.py --smoke      # 300-step learnability check, no training run
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# pxai's data pipeline applies these; B-cos wants raw [0,1], so we invert them.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BcosBackbone(nn.Module):
    """A B-cos network exposing pxai's backbone interface.

    Provides `.out_channels` and `forward(x) -> (B, C, H, W)`, so it drops into
    `pxai/models/__init__.py` beside the timm backbones. The heads (ProtoPNet, CBM,
    B-cos-style) attach unchanged.

    NOTE: mixing a B-cos backbone with a non-B-cos head breaks the exact-linearity
    property for the *head*, though not for the backbone. For the prediction in the
    module docstring, use it with the B-cos-style head, whose 1x1 conv preserves
    linearity.
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True,
                 b: float = 2.0):
        super().__init__()
        try:
            from bcos.models.pretrained import resnet18, resnet34, resnet50
            from bcos import modules  # noqa: F401  (import check)
        except ImportError as e:
            raise ImportError(
                "the `bcos` package is required:  pip install bcos\n"
                f"({e})") from e

        ctor = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
        if name not in ctor:
            raise ValueError(f"unsupported B-cos backbone {name!r}; "
                             f"choose from {sorted(ctor)}")
        net = ctor[name](pretrained=pretrained)

        # Strip the classifier: we want the feature map, not logits. B-cos ResNets
        # follow torchvision layout, so everything up to `layer4` is the feature trunk.
        self.trunk = nn.Sequential(*[m for n, m in net.named_children()
                                     if n not in ("fc", "avgpool", "logit_layer")])
        self.b = b
        self.register_buffer("_mean", _MEAN.clone())
        self.register_buffer("_std", _STD.clone())

        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224)
            self.out_channels = int(self.forward(probe).shape[1])

    def _encode(self, x):
        """3-channel ImageNet-normalised -> 6-channel [x, 1-x] in [0,1].

        Differentiable, so gradients flow back to the 3-channel input. The chain rule
        handles the [x, 1-x] pair automatically: d(1-x)/dx = -1, and autograd sums the
        two paths, which is exactly the attribution with respect to the original input.
        """
        x01 = (x * self._std + self._mean).clamp(0.0, 1.0)
        return torch.cat([x01, 1.0 - x01], dim=1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self._encode(x)
        return self.trunk(x)


# --------------------------------------------------------------------------- smoke
def _smoke():
    """300-step learnability check. Catches the failures that cost a night."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="resnet18")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    try:
        bb = BcosBackbone(a.name).to(dev)
    except ImportError as e:
        print(e)
        return
    print(f"{a.name}: out_channels={bb.out_channels}  "
          f"params={sum(p.numel() for p in bb.parameters())/1e6:.2f}M  device={dev}")

    x = torch.randn(2, 3, 224, 224, device=dev)
    f = bb(x)
    print(f"forward: (2,3,224,224) -> {tuple(f.shape)}   "
          f"grid {f.shape[-1]}x{f.shape[-1]}  stride {224 // f.shape[-1]}")

    # the property the whole exercise exists to test
    xi = x.clone().requires_grad_(True)
    out = bb(xi).mean()
    g, = torch.autograd.grad(out, xi)
    print(f"gradient wrt the 3-channel input: {tuple(g.shape)}  "
          f"|g| mean {g.abs().mean():.3e}   (non-zero => the encoding is differentiable)")

    head = nn.Linear(bb.out_channels, 11).to(dev)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(head.parameters()), lr=3e-4)
    y = torch.randint(0, 11, (8,), device=dev)
    xb = torch.rand(8, 3, 224, 224, device=dev)
    first = None
    for i in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(bb(xb).mean((-2, -1))), y)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    print(f"\noverfit 8 fixed samples: loss {first:.4f} -> {loss.item():.4f}")
    print("  PASS -- gradients flow, the wrapper trains" if loss.item() < first * 0.5
          else "  FAIL -- loss did not fall; check the trunk assembly")

    print("""
NEXT IF THIS PASSES
  1. register in pxai/models/__init__.py:
         from .bcos_backbone import BcosBackbone
         if cfg["backbone"]["name"].startswith("bcos_"):
             self.backbone = BcosBackbone(cfg["backbone"]["name"][5:])
             ch = self.backbone.out_channels
  2. config: backbone.name: bcos_resnet18, model.kind: bcos, batch_size 32
  3. preflight_learns.py, then train 3 seeds (~4h)
  4. THE TEST:
         python -u probe_gradattr.py --device cuda --runs roi477_bcosnet_120ep
     gain ~ 1.0x  -> B-cos's exact-linearity claim HOLDS, and Part II is validated by
                     a correct null prediction
     gain > 1     -> even a real B-cos has a read-out gap
""")


if __name__ == "__main__":
    _smoke()