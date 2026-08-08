"""
visualise_protograd.py — see the localizable ProtoPNet explanation.

Renders, per image:

    input | ours:protopnet | protograd | protograd_smooth | proto #1 | #2 | #3

`ours:protopnet` is the current corrected attribution (argmax cells of the 7x7
similarity map, learned weights, bilinearly upsampled). `protograd` is the gradient of
each prototype's pooled similarity with respect to the INPUT, at pixel resolution.
The last three columns are the three highest-weighted prototypes individually — the
"this looks like that" claim, one prototype at a time.

Measured seed-averaged over five seeds, 275 images:

    variant             mass   c@1%   1st hit
    ours:protopnet      3.08   3.68     3.7%
    protograd           5.46  11.80     0.0%
    protograd_smooth    5.62  12.77     0.0%
    per_prototype       6.24  12.37     0.1%     <- single prototypes localise BEST

WHAT TO LOOK FOR
  * ours:protopnet should show the familiar smooth blob or ring at 32px granularity —
    the 7x7 grid upsampled.
  * protograd should be visibly tighter and follow the egg's actual outline, because
    sim_pooled is a max over cells, so its gradient is non-zero only through the argmax
    cell's receptive field, resolved at pixel level within it.
  * the per-prototype columns are the real test. Each should sit on a specific,
    coherent region. If they all pile onto the same spot, the prototypes are redundant;
    if each finds a different structure (shell, plug, interior), the case-based claim is
    doing real work and is worth a figure in the thesis.

    python -u visualise_protograd.py --run roi477_protopnet_120ep --n 6 \
        --out figs/protograd_overview.png --device cuda

    # one class, to inspect a specific species
    python -u visualise_protograd.py --run roi477_protopnet_120ep \
        --only "Capillaria philippinensis" --n 5 \
        --out figs/protograd_capillaria.png --device cuda
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.eval.cropgeom import load_coco, box_in_crop

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(x):
    return np.clip(x.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def conc(a, mask):
    """Positive-mass concentration; 1.0 = no better than uniform."""
    a = np.clip(np.asarray(a, float), 0, None).ravel()
    m = np.asarray(mask, bool).ravel()
    area = m.mean()
    return float(a[m].sum() / a.sum() / area) if a.sum() > 0 and area > 0 else float("nan")


def class_w(head, target):
    w = head.last.weight
    if getattr(head, "pip_sparsity", False):
        w = F.relu(w)
    return w[target]


def baseline(model, x, target):
    head = model.head
    with torch.no_grad():
        _, sim = head._similarities(model.backbone(x))
        B, P, h, w = sim.shape
        wy = class_w(head, target).view(B, P, 1, 1)
        flat = sim.reshape(B, P, h * w)
        mx, idx = flat.max(-1)
        sp = torch.zeros_like(flat)
        sp.scatter_(2, idx.unsqueeze(-1), (mx * wy.view(B, P)).unsqueeze(-1))
        a = sp.view(B, P, h, w).sum(1, keepdim=True)
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


def protograd(model, x, target, protos=None, noise=0.0, n=1):
    """d sim_pooled[p] / dx at pixel resolution, weighted and summed over `protos`."""
    head = model.head
    with torch.no_grad():
        wy = class_w(head, target)                              # (B,P)
    if protos is None:
        protos = [p for p in range(wy.shape[1]) if float(wy[0, p]) > 0] or \
            list(range(wy.shape[1]))
    total = torch.zeros_like(x[:, :1])
    for _ in range(max(1, n)):
        xi = x if noise <= 0 else x + torch.randn_like(x) * noise
        xi = xi.clone().detach().requires_grad_(True)
        sp, _ = head._similarities(model.backbone(xi))
        g, = torch.autograd.grad((sp[:, protos] * wy[:, protos]).sum(), xi)
        total = total + (g * xi).abs().sum(1, keepdim=True).detach()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roi477_protopnet_120ep")
    ap.add_argument("--out", default="figs/protograd_overview.png")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--only", default=None, help="restrict to one class")
    ap.add_argument("--n-protos", type=int, default=3,
                    help="how many individual prototypes to render")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    a = ap.parse_args()

    cfg = load_config(f"configs/generated/{a.run}.yaml")
    cfg["device"] = a.device
    dev = pick_device(cfg["device"])
    S, root = cfg["data"]["img_size"], cfg["data"]["root"]
    lab = os.path.join(root, "labels.json")
    if not os.path.exists(lab):
        lab = os.path.join(os.path.dirname(root.rstrip("/")),
                           "Chula-ParasiteEgg-11", "labels.json")
    ann = load_coco(lab)
    loaders = build_loaders(cfg)
    classes = loaders.classes
    cfg["model"]["num_classes"] = len(classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(f"runs/{a.run}/best.pt", map_location=dev)["model"])
    model.eval()

    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))
    if a.only:
        if a.only not in classes:
            raise SystemExit(f"--only {a.only!r} not in {classes}")
        ci = classes.index(a.only)
        idxs = [i for i in idxs if base.samples[i][1] == ci]
    rng = np.random.default_rng(a.seed)
    pick = [idxs[j] for j in rng.choice(len(idxs), min(a.n, len(idxs)), replace=False)]

    cols = ["input", "ours:protopnet", "protograd", "protograd_smooth"] + \
           [f"proto #{i + 1}" for i in range(a.n_protos)]
    fig, axes = plt.subplots(len(pick), len(cols),
                             figsize=(2.25 * len(cols), 2.55 * len(pick)), squeeze=False)

    for r, gi in enumerate(pick):
        path, label = base.samples[gi]
        x, _ = base[gi]
        x = x.unsqueeze(0).to(dev)
        t = torch.tensor([label], device=dev)
        box = box_in_crop(path, ann, S, a.margin, True)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
            wy = class_w(model.head, t)[0]
        top = torch.topk(wy, a.n_protos).indices.tolist()

        maps = {"ours:protopnet": baseline(model, x, t),
                "protograd": protograd(model, x, t),
                "protograd_smooth": protograd(model, x, t, noise=0.10, n=8)}
        for i, p in enumerate(top):
            maps[f"proto #{i + 1}"] = protograd(model, x, t, protos=[p])

        img = denorm(x[0])
        for c, name in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(img)
            if name != "input":
                m = maps[name].detach().float().cpu().numpy()[0, 0]
                ax.imshow(norm01(m), cmap="inferno", alpha=0.5)
                cv = conc(m, box) if box is not None else float("nan")
                extra = f"  w={float(wy[top[int(name[-1]) - 1]]):.2f}" \
                    if name.startswith("proto #") else ""
                ax.set_title(f"c{cv:.1f}{extra}", fontsize=6.5)
            else:
                ok = "OK" if pred == label else f"-> {classes[pred][:11]}"
                ax.set_title(f"{classes[label][:17]}\n{ok}", fontsize=6.5)
            if box is not None:
                ys, xs = np.nonzero(box)
                ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                       fill=False, ec="lime", lw=1.3))
            ax.axis("off")
            if r == 0:
                ax.text(0.5, 1.24, name, transform=ax.transAxes, ha="center",
                        fontsize=8, weight="bold")

    fig.suptitle(f"{a.run}   green = annotation box   c = concentration "
                 f"(1.0 = uniform, 16.3 = perfect)   w = prototype weight",
                 fontsize=9, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=115, bbox_inches="tight")
    print(f"wrote {a.out}   {len(pick)} images x {len(cols)} panels")
    print("""
  ours:protopnet -> the 7x7 grid upsampled: smooth blobs at 32px granularity
  protograd      -> should hug the egg's actual outline
  proto #1..3    -> the case-based claim, one prototype at a time. Different regions
                    per prototype means the prototypes are doing distinct work; all on
                    the same spot means they are redundant.""")


if __name__ == "__main__":
    main()
