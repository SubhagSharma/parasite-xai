#!/usr/bin/env python
# apply_egg_push.py -- restrict prototype projection to patches that overlap the egg
"""
THE DEFECT
----------
pxai/train.py `_push_prototypes` snaps each prototype onto the nearest patch from a
same-class image. Class-constrained, but NOT location-constrained: the candidate set is
all h*w spatial positions of every class-c image, including pure background.

    patches = zf_img[img_mask].reshape(-1, D)     # ALL 49 patches per image
    d = ((patches - pv) ** 2).sum(1)
    md, mi = d.min(0)                             # nearest patch, wherever it is

prototype_sources.png shows the outcome: 21 of 24 prototypes are labelled "background".
The model's prototypes are learned pieces of background, so its explanations point at
background -- faithfully. The low conc_pos is a correct report of what the model uses.

Chen et al. do not need this constraint because CUB birds fill the frame. Here the egg
is 2-12% of the image and unconstrained push has nowhere to go but background.

THE FIX
-------
Restrict candidates to patches whose receptive field overlaps the annotation box. A
feature cell counts if ANY of its pixels fall inside the box (adaptive max-pool of the
box mask down to the feature grid). Enabled by `train.push_egg_only: true`; absent or
false reproduces the old behaviour exactly.

Falls back to unconstrained push for any prototype with no in-box candidate, and
reports the count -- a high fallback rate means the constraint is too tight and the
result must be qualified.

    python apply_egg_push.py --check | --revert
"""
import argparse, ast, os, shutil, sys

TARGET = "pxai/train.py"

OLD_SIG = '''def _push_prototypes(model, loader, device, max_batches: int | None = None):'''
NEW_SIG = '''def _push_prototypes(model, loader, device, max_batches: int | None = None,
                     egg_only: bool = False, ann=None, img_size: int = 224,
                     margin: float = 0.20):'''

OLD_CALL = '''            stats = _push_prototypes(model, loaders.train, device, max_batches=pmb)'''
NEW_CALL = '''            _egg = cfg["train"].get("push_egg_only", False)
            stats = _push_prototypes(
                model, loaders.train, device, max_batches=pmb,
                egg_only=_egg, ann=(_push_ann if _egg else None),
                img_size=cfg["data"]["img_size"])'''

OLD_LOOP = '''    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        z = head.add_on(model.features(x))                       # (B,D,H,W)
        B, _, H, W = z.shape
        # per-image patch bank so we can restrict by the image's class label
        zf_img = z.permute(0, 2, 3, 1).reshape(B, H * W, D)      # (B, HW, D)

        # for each class present in this batch, match only ITS prototypes to ITS patches
        for c in torch.unique(y).tolist():
            protos_c = proto_of_class[c]
            if not protos_c:
                continue
            img_mask = (y == c)
            patches = zf_img[img_mask].reshape(-1, D)            # (n_c*HW, D)
            if patches.numel() == 0:
                continue
            for p in protos_c:
                pv = head.prototypes[p].view(1, D)
                d = ((patches - pv) ** 2).sum(1)
                md, mi = d.min(0)
                if md < best_dist[p]:
                    best_dist[p] = md.cpu()
                    best_vec[p] = patches[mi].detach().cpu()'''

NEW_LOOP = '''    # egg_only: iterate the dataset by index so file paths are available for the
    # annotation lookup. The default loader yields (x, y) with no path.
    fell_back = 0
    if egg_only:
        from torch.utils.data import Subset
        from .eval.cropgeom import box_in_crop
        ds = loader.dataset
        _b = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(_b, Subset):
            _b = _b.dataset
        _idx = list(ds.indices) if isinstance(ds, Subset) else list(range(len(_b.samples)))
        bs = loader.batch_size or 32
        chunks = [_idx[i:i + bs] for i in range(0, len(_idx), bs)]
        if max_batches is not None:
            chunks = chunks[:max_batches]
        stream = (( torch.stack([_b[i][0] for i in ch]),
                    torch.tensor([_b.samples[i][1] for i in ch]),
                    [_b.samples[i][0] for i in ch] ) for ch in chunks)
    else:
        stream = ((x, y, None) for x, y in loader)

    for bi, (x, y, paths) in enumerate(stream):
        if max_batches is not None and bi >= max_batches and not egg_only:
            break
        x = x.to(device)
        y = y.to(device)
        z = head.add_on(model.features(x))                       # (B,D,H,W)
        B, _, H, W = z.shape
        zf_img = z.permute(0, 2, 3, 1).reshape(B, H * W, D)      # (B, HW, D)

        # (B, HW) bool: does this feature cell overlap the annotated egg?
        if egg_only:
            keep = torch.zeros(B, H * W, dtype=torch.bool)
            for bi_, pth in enumerate(paths):
                bm = box_in_crop(pth, ann, img_size, margin, True)
                if bm is None or not bm.any():
                    continue
                t = torch.from_numpy(bm.astype("float32"))[None, None]
                cell = F.adaptive_max_pool2d(t, (H, W))[0, 0] > 0
                keep[bi_] = cell.reshape(-1)
        else:
            keep = torch.ones(B, H * W, dtype=torch.bool)

        for c in torch.unique(y).tolist():
            protos_c = proto_of_class[c]
            if not protos_c:
                continue
            img_mask = (y == c)
            patches = zf_img[img_mask].reshape(-1, D)            # (n_c*HW, D)
            if patches.numel() == 0:
                continue
            kmask = keep[img_mask.cpu()].reshape(-1).to(patches.device)
            cand = patches[kmask] if kmask.any() else patches
            if not kmask.any():
                fell_back += 1
            for p in protos_c:
                pv = head.prototypes[p].view(1, D)
                d = ((cand - pv) ** 2).sum(1)
                md, mi = d.min(0)
                if md < best_dist[p]:
                    best_dist[p] = md.cpu()
                    best_vec[p] = cand[mi].detach().cpu()'''

OLD_RET = '''    return {"pushed": int((~missed).sum()), "missed": int(missed.sum())}'''
NEW_RET = '''    return {"pushed": int((~missed).sum()), "missed": int(missed.sum()),
            "egg_only": bool(egg_only), "fell_back": int(fell_back)}'''

OLD_HDR = '''    kind = cfg["model"]["kind"]'''
NEW_HDR = '''    kind = cfg["model"]["kind"]
    # annotations for egg-constrained push; loaded once, not per push
    _push_ann = None
    if kind == "protopnet" and cfg["train"].get("push_egg_only", False):
        import os as _os
        from .eval.cropgeom import load_coco as _load_coco
        _r = cfg["data"]["root"]
        _lp = _os.path.join(_r, "labels.json")
        if not _os.path.exists(_lp):
            _lp = _os.path.join(_os.path.dirname(_r.rstrip("/")),
                                "Chula-ParasiteEgg-11", "labels.json")
        _push_ann = _load_coco(_lp)
        print(f"[push] egg-constrained projection ON, annotations from {_lp}",
              flush=True)'''

EDITS = [("push signature", OLD_SIG, NEW_SIG),
         ("push call site", OLD_CALL, NEW_CALL),
         ("push candidate loop", OLD_LOOP, NEW_LOOP),
         ("push return dict", OLD_RET, NEW_RET),
         ("annotation load", OLD_HDR, NEW_HDR)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = TARGET + ".bak-eggpush"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TARGET); print(f"restored {TARGET}"); return

    src = open(TARGET).read()
    if "push_egg_only" in src:
        sys.exit("already patched. --revert first to redo.")

    out, bad = src, []
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            bad.append((name, n, old)); print(f"  {'MISS' if n==0 else 'AMBIG'}  {name} ({n})")
            continue
        out = out.replace(old, new, 1); print(f"  ok    {name}")
    if bad:
        print(f"\n{len(bad)} edit(s) failed. Nothing written.")
        for name, n, old in bad:
            print(f"\n--- {name}, expected ---\n{old}")
        sys.exit(1)
    if "import torch.nn.functional as F" not in out:
        out = out.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
        print("  ok    added torch.nn.functional import")
    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"\nwould not parse: {e}\nNothing written.")
    print("  parses OK")
    if a.check:
        print("\n--check: nothing written."); return
    shutil.copy2(TARGET, bak); open(TARGET, "w").write(out)
    print(f"\nbackup -> {bak}\npatched -> {TARGET}")


if __name__ == "__main__":
    main()
