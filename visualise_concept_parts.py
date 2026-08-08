"""
visualise_concept_parts.py — the attention maps, which ARE the explanation.

WHAT MAKES THIS DIFFERENT FROM EVERYTHING ELSE IN THE PROJECT
-------------------------------------------------------------
Every other explanation here is a reconstruction: ProtoPNet's similarity map upsampled
from 7x7, Grad-CAM's gradient-weighted feature map upsampled, the Part II gradient
attribution computed by a backward pass. Each is an artefact produced *about* the model.

ConceptPartHead's attention map is the model's own computation. Concept k is predicted
from a scalar projection of slot k alone, so the ONLY route to that concept's value runs
through attention map k. The map is not an explanation of the concept -- it is where the
concept was measured.

And unlike a prototype, it has a NAME. `A_operculum` is not "prototype 3, which appears
to be some kind of edge". It is the operculum, as defined by CDC DPDx.

WHAT TO LOOK FOR
  * a concept map should sit on the structure it names, and on the SAME structure
    across images of a species
  * concepts with strong class contrast should localise best. From concepts_v3.csv:
        contents=unembryonated   5/11 classes   most contrast
        operculum                ~3/11
        has_polar_plugs          ~2/11
        symmetry=symmetric      10/11           almost none -- expected to fail, and
                                                the supervised CBM already shows it
                                                failing at TPR 0.600
  * maps within a family (shell_texture=smooth vs =striated) SHOULD overlap; they
    describe the same anatomy. Maps across families should not.

    python -u visualise_concept_parts.py --run roi477_parts_120ep --device cuda \\
        --out figs/parts_overview.png

    # one species, every concept it possesses
    python -u visualise_concept_parts.py --run roi477_parts_120ep --device cuda \\
        --only "Trichuris trichiura" --out figs/parts_trichuris.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.concepts_loader import load_concept_table
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
    ar = m.mean()
    return float(a[m].sum() / a.sum() / ar) if a.sum() > 0 and ar > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roi477_parts_120ep")
    ap.add_argument("--out", default="figs/parts_overview.png")
    ap.add_argument("--only", default=None, help="restrict to one species")
    ap.add_argument("--n", type=int, default=4, help="images per figure")
    ap.add_argument("--top-k", type=int, default=6, help="concepts to render")
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
    head = model.head

    csv_path = cfg["model"]["concept_parts"]["concepts_csv"]
    table, names = load_concept_table(csv_path, classes)

    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))
    if a.only:
        if a.only not in classes:
            raise SystemExit(f"--only {a.only!r} not in {classes}")
        idxs = [i for i in idxs if base.samples[i][1] == classes.index(a.only)]
    rng = np.random.default_rng(a.seed)
    pick = [idxs[j] for j in rng.choice(len(idxs), min(a.n, len(idxs)), replace=False)]

    # choose concepts by CLASS CONTRAST: a concept present in ~half the classes carries
    # the most discriminative signal, and the head can only localise what discriminates
    npos = table.sum(0).numpy()
    contrast = np.minimum(npos, len(classes) - npos)
    ks = list(np.argsort(-contrast)[:a.top_k])
    print(f"concepts by class contrast: "
          f"{[(names[k], int(npos[k])) for k in ks]}")

    cols = 1 + len(ks)
    fig, axes = plt.subplots(len(pick), cols,
                             figsize=(2.2 * cols, 2.5 * len(pick)), squeeze=False)

    for r, gi in enumerate(pick):
        path, label = base.samples[gi]
        x, _ = base[gi]
        x = x.unsqueeze(0).to(dev)
        box = box_in_crop(path, ann, S, a.margin, True)
        with torch.no_grad():
            feat = model.features(x)
            logits, c_logit = head(feat)
            pred = logits.argmax(1).item()
            cprob = torch.sigmoid(c_logit)[0]
        img = denorm(x[0])

        ax = axes[r][0]
        ax.imshow(img)
        if box is not None:
            ys, xs = np.nonzero(box)
            ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                   fill=False, ec="lime", lw=1.4))
        ax.set_title(f"{classes[label][:16]}\n"
                     f"{'OK' if pred == label else '-> ' + classes[pred][:11]}",
                     fontsize=6.5)
        ax.axis("off")
        if r == 0:
            ax.text(0.5, 1.24, "input", transform=ax.transAxes, ha="center",
                    fontsize=8, weight="bold")

        for c, k in enumerate(ks):
            ax = axes[r][c + 1]
            m = head.concept_map(feat, int(k), size=x.shape[-2:])
            arr = m.detach().float().cpu().numpy()[0, 0]
            ax.imshow(img)
            ax.imshow(norm01(arr), cmap="inferno", alpha=0.5)
            if box is not None:
                ys, xs = np.nonzero(box)
                ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                       fill=False, ec="lime", lw=1.1))
            cv = conc(arr, box) if box is not None else float("nan")
            truth = int(table[label, k])
            ax.set_title(f"c{cv:.1f}  p={float(cprob[k]):.2f}"
                         f"{' *' if truth else ''}", fontsize=6)
            ax.axis("off")
            if r == 0:
                ax.text(0.5, 1.24, names[int(k)][:20], transform=ax.transAxes,
                        ha="center", fontsize=7, weight="bold")

    fig.suptitle(f"{a.run}   green = annotation box   c = concentration "
                 f"(1.0 = uniform)   p = predicted concept   * = concept is TRUE "
                 f"for this species", fontsize=8, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("""
READING
  Each column is ONE NAMED CONCEPT's attention map. It is not a reconstruction: the
  concept is predicted from a projection of that slot alone, so the map is where the
  concept was measured.

  c   concentration on the annotation box. 1.0 = uniform, higher is tighter.
  p   the predicted concept probability.
  *   the concept is TRUE for this species per DPDx.

  A map with p high AND * present AND c high is a concept that was found, in the right
  place, on a species that has it. A map with p high and no * is the model asserting a
  feature the species does not possess.""")


if __name__ == "__main__":
    main()
