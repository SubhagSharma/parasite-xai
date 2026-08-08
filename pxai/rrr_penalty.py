"""Right-for-the-Right-Reasons regularisation — train the model to stop using the
background, then test whether the localisation/faithfulness trade-off closes.

===============================================================================
WHY
===============================================================================
Part II measured a trade-off across all three ante-hoc heads:

    head        localisation (c@1%)          faithfulness (deletion)
    protopnet   2.26 -> 12.19  5.4x BETTER   0.0166 -> 0.1333  8.0x WORSE
    cbm         2.10 -> 11.65  5.6x BETTER   0.0290 -> 0.1440  5.0x WORSE
    bcos        5.72 -> 12.59  2.2x BETTER   0.0465 -> 0.0963  2.1x WORSE

The explanation offered in Part II SEC 5.5.1 is that the two metrics measure different
things and only agree when the model depends on the object:

    deletion  rewards finding the pixels the model ACTUALLY USES  -> background
    conc_+    rewards finding the pixels ON THE OBJECT            -> the egg

That is an interpretation, not a measurement. **This experiment tests it.**

If the trade-off is caused by the shortcut, then suppressing the shortcut should CLOSE
it. Train a model that cannot use the background, and the two explanations should
converge — because there is no longer a gap between "what the model uses" and "where
the object is".

    PREDICTION, RECORDED BEFORE THE RUN
      deletion ratio (gradattr / native) should fall from 8.0x toward 1.0x
      egg-masked accuracy should fall from 0.2485 toward chance (0.0909)
      conc_+ of the NATIVE explanation should rise from 2.53

    If the ratio does not move, the SEC 5.5.1 interpretation is wrong and the trade-off
    has some other cause.

===============================================================================
PRIOR WORK — THIS METHOD IS NOT NOVEL
===============================================================================
**Ross, Hughes & Doshi-Velez (2017), "Right for the Right Reasons", IJCAI.** Defines an
annotation matrix A marking which input dimensions should be irrelevant, and penalises
the input gradient there. Their stated reason for shrinking irrelevant gradients rather
than enlarging relevant ones applies directly here: gradients for relevant inputs should
be small far from the decision boundary, and the right magnitude is not known in advance.

**Li et al. (2018), "Tell Me Where to Look" (GAIN), CVPR.** The masking variant: two
parameter-sharing streams, one finding the regions that support recognition and one
checking that all of them have been found. `GAINext` uses external supervision and is
explicitly framed as making features robust to dataset bias.

**What is new here is not the loss.** It is using it as an instrument: the project has a
*measured, control-validated* shortcut (displaced-mask control puts 99.7-99.9% of the
occlusion drop on the egg's absence) and a *measured* localisation/faithfulness gap.
Neither prior paper reports both axes, so neither can test whether suppressing the
shortcut closes the gap. That prediction is the contribution; RRR is the tool.

===============================================================================
THE LOSS
===============================================================================
    L = L_task  +  lambda * mean_over_pixels_outside_the_box( (d logit_y / dx)^2 )

Head-agnostic: it constrains d(class logit)/dx, which every head has. No change to
ProtoPNet, CBM or B-cos internals, so `protopnet`, `cbm` and `bcos` arms are directly
comparable and the existing checkpoints remain the controls.

Two implementation points that matter:

  * **create_graph=True.** The penalty is itself a function of a gradient, so the
    backward pass must differentiate through it. Without this the term is a constant and
    silently does nothing — which is exactly the class of bug that has cost this project
    several nights.
  * **fp32.** Double backward under autocast is numerically fragile; the penalty is
    computed outside autocast.

Cost: one extra backward per step, so roughly 1.5-1.8x training time. Cheaper than GAIN,
which needs a second forward stream.

===============================================================================
CONFIG
===============================================================================
    train:
      rrr_lambda: 1.0          # 0 disables; the loss is then bit-identical to before

lambda is a starting point, not tuned. Ross et al. note it should be set so the
"right answers" and "right reasons" terms are the same order of magnitude — check the
printed ratio in the first epoch and adjust.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class RRRPenalty:
    """Penalise input-gradient magnitude outside the annotation box.

    Masks are built once per image path and cached: box_in_crop does PIL work and would
    otherwise dominate the step time.
    """

    def __init__(self, ann, img_size: int, margin: float = 0.20,
                 lam: float = 1.0, square: bool = True):
        self.ann = ann
        self.S = img_size
        self.margin = margin
        self.lam = lam
        self.square = square
        self._cache: dict = {}
        self.n_hit = 0
        self.n_miss = 0

    def outside_mask(self, paths, device) -> Optional[torch.Tensor]:
        """(B,1,S,S) float, 1 outside the egg box and 0 inside. None if no boxes."""
        from pxai.eval.cropgeom import box_in_crop
        out = []
        for p in paths:
            if p not in self._cache:
                m = box_in_crop(p, self.ann, self.S, self.margin, self.square)
                self._cache[p] = None if (m is None or not m.any()) else \
                    torch.from_numpy((~m).astype(np.float32))
            m = self._cache[p]
            if m is None:
                self.n_miss += 1
                out.append(torch.ones(self.S, self.S))
            else:
                self.n_hit += 1
                out.append(m)
        if not out:
            return None
        return torch.stack(out).unsqueeze(1).to(device)

    def __call__(self, model, x, y, paths) -> torch.Tensor:
        """The penalty term. Returns a 0-d tensor; zero when lam <= 0."""
        if self.lam <= 0:
            return x.new_zeros(())
        mask = self.outside_mask(paths, x.device)
        if mask is None:
            return x.new_zeros(())

        # fp32 and create_graph: the penalty is a function of a gradient, so the
        # optimiser must be able to differentiate through it.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xi = x.float().clone().detach().requires_grad_(True)
            out = model(xi)
            out = out[0] if isinstance(out, tuple) else out
            sel = out.gather(1, y.view(-1, 1)).sum()
            g, = torch.autograd.grad(sel, xi, create_graph=True)
            # mean over masked pixels, not sum: keeps lambda comparable across images
            # with different box sizes
            pen = ((g ** 2).sum(1, keepdim=True) * mask).sum() / mask.sum().clamp_min(1)
        return self.lam * pen


def build_rrr(cfg, classes=None) -> Optional[RRRPenalty]:
    """Construct from config, or None when `train.rrr_lambda` is absent or 0."""
    lam = float(cfg.get("train", {}).get("rrr_lambda", 0.0))
    if lam <= 0:
        return None
    from pxai.eval.cropgeom import load_coco
    root = cfg["data"]["root"]
    lp = os.path.join(root, "labels.json")
    if not os.path.exists(lp):
        lp = os.path.join(os.path.dirname(root.rstrip("/")),
                          "Chula-ParasiteEgg-11", "labels.json")
    print(f"[rrr] explanation regularisation ON, lambda={lam}, boxes from {lp}",
          flush=True)
    return RRRPenalty(load_coco(lp), cfg["data"]["img_size"], lam=lam)


# --------------------------------------------------------------- indexed loader
class _WithPath(torch.utils.data.Dataset):
    """Wrap a dataset so __getitem__ also returns the source file path.

    The training loop is `for x, y in loaders.train` and has no path, but the box
    lookup needs one. Rather than change that contract for every head, this wrapper is
    used ONLY when rrr_lambda > 0, so the default path stays byte-identical.
    """

    def __init__(self, ds):
        from torch.utils.data import Subset
        self.ds = ds
        base = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(base, Subset):
            base = base.dataset
        self.base = base
        self.idx = list(ds.indices) if isinstance(ds, Subset) \
            else list(range(len(base.samples)))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, self.base.samples[self.idx[i]][0]


def make_indexed_loader(loader):
    """A DataLoader matching `loader` but yielding (x, y, path)."""
    from torch.utils.data import DataLoader
    return DataLoader(
        _WithPath(loader.dataset),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=getattr(loader, "num_workers", 4),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=getattr(loader, "drop_last", False),
    )
