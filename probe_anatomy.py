"""
probe_anatomy.py — is the region a concept slot attends to actually the named structure?

THE ONE OPEN QUESTION
=====================
Six of seven requirements for the original goal are met:

    one slot per named DPDx feature                     23 slots, named from the table
    each predicts its feature                           0.9882 macro, 22/23 at ceiling
    each produces a spatial map                         attention map per concept
    maps land on the egg                                4.83-6.03 vs uniform 1.0
    maps land on the same spot across images            centroid spread 0.148
    the concepts DRIVE the prediction                   88% targeted intervention
    maps land on the NAMED STRUCTURE                    <- unverified

Everything measured so far shows the slots find *consistent, causal, well-localised*
regions. Nothing yet shows that `A_operculum` is on the operculum rather than on some
other region that happens to correlate with it.

THREE TESTS, IN INCREASING STRENGTH
===================================

--- 1. CROSS-SPECIES CONSISTENCY (no external knowledge required) ---------------
`operculum` is true for 3 species with different egg shapes and sizes. If its centroid
lands at the same BOX-RELATIVE position on all three, the slot has found something
anatomical rather than a species-specific artefact.

    within-species spread already measured at 0.148. If BETWEEN-species spread is
    comparable, the location is a property of the structure. If it is much larger, each
    species has its own idiosyncratic region and the shared name means nothing.

This is the strongest test available without an anatomist.

--- 2. ANATOMICAL POSITION PRIOR (weak external knowledge) ----------------------
Some DPDx features have known positions on an egg, and they differ from each other:

    operculum, has_polar_plugs,
    has_polar_filaments, has_polar_knob   POLAR   -- at an end, so |y - 0.5| is large
    contents=*                            CENTRAL -- the interior
    shell_texture=*, shell_thickness=*    PERIPHERAL -- the wall, so radius is large

These are coarse, but they are independent of the model and they discriminate: a slot
attending to the centre cannot be finding a polar plug. Testing whether the three groups
separate on |y - 0.5| and on radius from centre gives quantitative evidence without an
expert.

    groups separate as predicted -> the slots respect anatomy
    groups indistinguishable     -> the maps are consistent but not anatomical, and the
                                    claim must stop at "consistent evidence region"

--- 3. EXPERT REVIEW FIGURES (the test that finally matters) --------------------
Renders one figure per (concept, species) pair for the concepts with clear anatomical
definitions: five images, the attention map, the annotation box. A parasitologist can
answer "is this the operculum?" in a few minutes per figure.

This is the test CUB-based part discovery cannot run: bird keypoints have no clinical
definition, so "part 3 is consistent" is the strongest claim available there. Parasite
morphology is named in a diagnostic reference, so a discovered part is either right or
wrong.

    python -u probe_anatomy.py --run roi477_parts_120ep --device cuda
    python -u probe_anatomy.py --run roi477_parts_120ep --device cuda --figures
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics as st

import numpy as np
import torch
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.concepts_loader import load_concept_table
from pxai.eval.cropgeom import load_coco, box_in_crop

# Coarse anatomical priors from CDC DPDx. Deliberately weak -- they only need to
# DISCRIMINATE between groups, not to specify exact positions.
POLAR = ("operculum", "has_polar_plugs", "has_polar_filaments", "has_polar_knob")
CENTRAL = ("contents",)
PERIPHERAL = ("shell_texture", "shell_thickness")


def group_of(name: str) -> str:
    base = name.split("=")[0]
    if base in POLAR:
        return "polar"
    if base in CENTRAL:
        return "central"
    if base in PERIPHERAL:
        return "peripheral"
    return "other"


def centroid_in_box(a, mask):
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    if a.sum() <= 0:
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y1 <= y0 or x1 <= x0:
        return None
    H, W = a.shape
    gy, gx = np.mgrid[0:H, 0:W]
    cy = float((a * gy).sum() / a.sum())
    cx = float((a * gx).sum() / a.sum())
    return ((cy - y0) / (y1 - y0), (cx - x0) / (x1 - x0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roi477_parts_120ep")
    ap.add_argument("--n", type=int, default=10, help="images per class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--figures", action="store_true",
                    help="also render the expert-review figures (test 3)")
    ap.add_argument("--figdir", default="figs/expert_review")
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
    table, names = load_concept_table(
        cfg["model"]["concept_parts"]["concepts_csv"], classes)
    K = len(names)

    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))
    rng = np.random.default_rng(a.seed)

    # (concept, class) -> list of box-relative centroids, only where the concept is TRUE
    cent = collections.defaultdict(list)
    samples = collections.defaultdict(list)          # for the figures

    for ci in range(len(classes)):
        pool = [i for i in idxs if base.samples[i][1] == ci]
        if not pool:
            continue
        for gi in [pool[j] for j in rng.choice(len(pool), min(a.n, len(pool)),
                                               replace=False)]:
            path, y = base.samples[gi]
            x, _ = base[gi]
            x = x.unsqueeze(0).to(dev)
            box = box_in_crop(path, ann, S, a.margin, True)
            if box is None or not box.any():
                continue
            with torch.no_grad():
                feat = model.features(x)
            for k in range(K):
                if int(table[y, k]) != 1:
                    continue
                m = head.concept_map(feat, k, size=x.shape[-2:])
                arr = m.detach().float().cpu().numpy()[0, 0]
                c = centroid_in_box(arr, box)
                if c:
                    cent[(k, ci)].append(c)
                    if len(samples[(k, ci)]) < 5:
                        samples[(k, ci)].append((gi, arr, box))

    # ============================================ 1. cross-species consistency
    print("=== 1. CROSS-SPECIES CONSISTENCY ===")
    print("  a concept true for several species: does it land in the same box-relative")
    print("  place on all of them? within-species spread is 0.148 for reference\n")
    print(f"  {'concept':<32}{'species':>8}{'within':>9}{'between':>9}{'verdict':>14}")
    multi = []
    for k in range(K):
        cs = [ci for ci in range(len(classes)) if (k, ci) in cent
              and len(cent[(k, ci)]) >= 4]
        if len(cs) < 2:
            continue
        within = st.mean([float(np.array(cent[(k, ci)]).std(0).mean()) for ci in cs])
        means = np.array([np.array(cent[(k, ci)]).mean(0) for ci in cs])
        between = float(means.std(0).mean())
        v = "ANATOMICAL" if between <= within * 1.5 else "species-specific"
        multi.append((names[k], len(cs), within, between, v))
    for nm, nc, w, b, v in sorted(multi, key=lambda r: r[3]):
        print(f"  {nm[:30]:<32}{nc:>8}{w:>9.3f}{b:>9.3f}{v:>14}")
    if multi:
        na = sum(1 for r in multi if r[4] == "ANATOMICAL")
        print(f"\n  {na}/{len(multi)} concepts land in a consistent place ACROSS species.")
        print("  between <= 1.5x within -> the location is a property of the STRUCTURE,")
        print("  not of the species. That is the strongest evidence obtainable without")
        print("  an anatomist.")

    # ================================================ 2. anatomical position prior
    print("\n=== 2. ANATOMICAL POSITION PRIOR ===")
    print("  DPDx says opercula and plugs are POLAR, contents CENTRAL, texture and")
    print("  thickness PERIPHERAL. Do the groups separate?\n")
    g = collections.defaultdict(lambda: {"polarity": [], "radius": []})
    for (k, ci), pts in cent.items():
        if len(pts) < 4:
            continue
        arr = np.array(pts)
        cy, cx = arr.mean(0)
        g[group_of(names[k])]["polarity"].append(abs(cy - 0.5))
        g[group_of(names[k])]["radius"].append(
            float(np.hypot(cy - 0.5, cx - 0.5)))
    print(f"  {'group':<14}{'|y-0.5|':>10}{'radius':>9}{'n':>5}   expectation")
    exp = {"polar": "high polarity", "central": "low radius",
           "peripheral": "high radius", "other": "-"}
    for name in ("polar", "central", "peripheral", "other"):
        if name not in g or not g[name]["polarity"]:
            continue
        print(f"  {name:<14}{st.mean(g[name]['polarity']):>10.3f}"
              f"{st.mean(g[name]['radius']):>9.3f}{len(g[name]['polarity']):>5}"
              f"   {exp[name]}")
    if "polar" in g and "central" in g and g["polar"]["polarity"]:
        dp = st.mean(g["polar"]["polarity"]) - st.mean(g["central"]["polarity"])
        print(f"\n  polar minus central polarity: {dp:+.3f}")
        print("  positive and sizeable -> the slots respect anatomy")
        print("  near zero -> maps are consistent but NOT anatomical; the claim must")
        print("  stop at 'consistent evidence region'")

    # ================================================== 3. expert review figures
    if a.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])
        os.makedirs(a.figdir, exist_ok=True)
        wanted = [k for k in range(K) if group_of(names[k]) in ("polar", "central")]
        made = 0
        for k in wanted:
            for ci in range(len(classes)):
                sm = samples.get((k, ci), [])
                if len(sm) < 3:
                    continue
                fig, axes = plt.subplots(1, len(sm), figsize=(2.4 * len(sm), 2.7),
                                         squeeze=False)
                for j, (gi, arr, box) in enumerate(sm):
                    x, _ = base[gi]
                    img = np.clip(x.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
                    ax = axes[0][j]
                    ax.imshow(img)
                    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
                    ax.imshow(np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1),
                              cmap="inferno", alpha=0.5)
                    ys, xs = np.nonzero(box)
                    ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                           fill=False, ec="lime", lw=1.2))
                    ax.axis("off")
                fig.suptitle(f"Is the highlighted region the "
                             f"**{names[k]}**?   species: {classes[ci]}",
                             fontsize=10)
                fig.tight_layout(rect=[0, 0, 1, 0.90])
                safe = f"{names[k].replace('=', '_')}__{classes[ci].replace(' ', '_')}"
                fig.savefig(os.path.join(a.figdir, f"{safe}.png"), dpi=110,
                            bbox_inches="tight")
                plt.close(fig)
                made += 1
        print(f"\n=== 3. EXPERT REVIEW ===")
        print(f"  {made} figures -> {a.figdir}/")
        print("  One question per figure, answerable in under a minute by a")
        print("  parasitologist. This is the test CUB-based part discovery cannot run:")
        print("  bird keypoints have no clinical definition, so 'part 3 is consistent'")
        print("  is the strongest claim available there. Parasite morphology is named")
        print("  in a diagnostic reference, so a discovered part is right or wrong.")


if __name__ == "__main__":
    main()
