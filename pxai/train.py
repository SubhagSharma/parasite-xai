"""Train an interpretable model (or the heavy black box) on parasite microscopy.

    python -m pxai.train --config configs/default.yaml

The loss adapts to the head:
  protopnet -> CE + cluster/separation + L1 sparsity (PIP-Net compactness),
               with periodic prototype projection (push) onto nearest patches.
  cbm       -> CE + concept BCE (when concept labels exist).
  bcos/bb   -> CE.
"""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import load_config, set_seed, pick_device, ensure_dir
from .data import build_loaders
from .models import build_model
from .rrr_penalty import build_rrr, make_indexed_loader
from .models.cbm import concept_loss
from .concepts_loader import (load_concept_table, concept_pos_weight,
                              concept_targets, concept_report,
                              print_concept_report)


def train(cfg):
    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    kind = cfg["model"]["kind"]
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
              flush=True)
    # Class-level concept supervision. Absent csv -> unsupervised bottleneck, i.e.
    # the previous behaviour, but now it says so instead of failing silently.
    ctable = cnames = cweight = None
    if kind in ("cbm", "concept_parts", "family_parts"):
        cpath = cfg["model"].get(kind, cfg["model"].get("cbm", {})).get("concepts_csv")
        if cpath:
            ctable, cnames = load_concept_table(cpath, loaders.classes)
            cweight = concept_pos_weight(ctable).to(device)
            ctable = ctable.to(device)
            k = ctable.shape[1]
            if k != cfg["model"]["cbm"]["num_concepts"]:
                raise ValueError(
                    f"{cpath} encodes {k} concepts but config says "
                    f"{cfg['model']['cbm']['num_concepts']}. Set num_concepts: {k}.")
            print(f"[cbm] {k} class-level concepts from {cpath}; "
                  f"pos_weight {cweight.min():.1f}-{cweight.max():.1f}", flush=True)
        else:
            print("[cbm] no model.cbm.concepts_csv -> bottleneck is UNSUPERVISED. "
                  "Accuracy is valid; interpretability claims are not.", flush=True)
    # Right-for-the-Right-Reasons: penalise input-gradient magnitude outside the
    # annotation box. Returns None (and costs nothing) when train.rrr_lambda is 0.
    rrr = build_rrr(cfg)
    train_loader = make_indexed_loader(loaders.train) if rrr is not None \
        else loaders.train
    best = 0.0
    out = ensure_dir(cfg["output_dir"])

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for _batch in tqdm(train_loader, desc=f"ep{epoch}", leave=False):
            if rrr is not None:
                x, y, _paths = _batch
            else:
                x, y = _batch
                _paths = None
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = _compute_loss(model, kind, x, y, ctable, cweight)
            if rrr is not None:
                # outside autocast: the penalty is a function of a gradient, and double
                # backward under fp16 is numerically fragile
                _pen = rrr(model, x, y, _paths)
                if epoch == 0 and rrr.n_hit + rrr.n_miss < 20 * x.shape[0]:
                    print(f"  [rrr] task {float(loss):.4f}  penalty {float(_pen):.4f}  "
                          f"ratio {float(_pen) / max(float(loss), 1e-9):.2f}", flush=True)
                loss = loss + _pen
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        if kind in ("protopnet", "protopnet_diverse", "protopnet_ms") \
                and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:
            # None = search the full training set (paper protocol). Set
            # train.push_max_batches in the config only to deliberately subsample.
            pmb = cfg["train"].get("push_max_batches", None)
            _egg = cfg["train"].get("push_egg_only", False)
            if kind == "protopnet_ms":
                from .models import push_multiscale
                stats = push_multiscale(model, loaders.train, device, max_batches=pmb)
            else:
                stats = _push_prototypes(
                model, loaders.train, device, max_batches=pmb,
                egg_only=_egg, ann=(_push_ann if _egg else None),
                img_size=cfg["data"]["img_size"])
            print(f"[push] epoch {epoch+1}: projected {stats['pushed']}/{model.head.num_protos} "
                  f"prototypes (coverage={'full' if pmb is None else f'{pmb} batches'})", flush=True)

        acc = evaluate_acc(model, loaders.val, device)
        print(f"epoch {epoch}: val_acc={acc:.4f}")
        if ctable is not None and (epoch + 1) % 10 == 0:
            model.eval()
            cl, ct = [], []
            with torch.no_grad():
                for xb, yb in loaders.val:
                    _, cg = model.head(model.features(xb.to(device)))
                    cl.append(cg.float().cpu())
                    ct.append(concept_targets(ctable, yb.to(device)).cpu())
            rows, mb, mr = concept_report(torch.cat(cl), torch.cat(ct), cnames)
            print_concept_report(rows, mb, mr, class_acc=acc)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": loaders.classes,
                        "cfg": cfg}, f"{out}/best.pt")
    print(f"best val acc: {best:.4f}  -> {out}/best.pt")
    return model


def _compute_loss(model, kind, x, y, ctable=None, cweight=None):
    if kind == "sea":
        from .models.sea import sea_loss
        return sea_loss(model, x, y, getattr(model, "loss_w", None))[0]
    if kind == "protopnet":
        feat = model.features(x)
        logits = model.head(feat)
        ce = F.cross_entropy(logits, y)
        cluster, sep = model.head.cluster_sep_cost(feat, y)
        l1 = model.head.l1_last()
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1
    if kind == "protopnet_ms":
        from .models import pyramid_forward
        feats = pyramid_forward(model.backbone, x)
        ce = F.cross_entropy(model.head(feats), y)
        cluster, sep = model.head.cluster_sep_cost(feats, y)
        # identical weights to the protopnet branch, so any difference is the head
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * model.head.l1_last()
    if kind == "protopnet_diverse":
        feat = model.features(x)
        logits = model.head(feat)
        ce = F.cross_entropy(logits, y)
        cluster, sep = model.head.cluster_sep_cost(feat, y)
        l1 = model.head.l1_last()
        # identical to the protopnet branch, plus the same-class diversity penalty.
        # diversity_cost() returns exactly 0 when w_orth and w_sparse are both 0, so an
        # unconfigured diverse head trains identically to the original.
        div = model.head.diversity_cost(feat, y)
        if getattr(model.head, "_last_usage", None) is not None:
            import math as _m
            _b = _m.log(model.head.ppc)
            if not hasattr(model.head, "_usage_printed"):
                print(f"  [usage] entropy {model.head._last_usage:.3f} / {_b:.3f} "
                      f"({model.head._last_usage / _b:.0%} balanced; 100% = every "
                      f"prototype wins equally)", flush=True)
                model.head._usage_printed = True
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1 + div
    if kind == "family_parts":
        # same concept supervision as concept_parts, so the ONLY difference is
        # family-shared attention (+ the optional polar prior)
        feat = model.features(x)
        logits, c_logit = model.head(feat)
        c_target = concept_targets(ctable, y) if ctable is not None else None
        return (F.cross_entropy(logits, y)
                + concept_loss(c_logit, c_target, pos_weight=cweight)
                + model.head.part_loss(feat))
    if kind == "concept_parts":
        # identical concept supervision to the CBM branch, so the two are
        # directly comparable; the only difference is that each concept is
        # predicted from its OWN attention slot, not from a global average
        feat = model.features(x)
        logits, c_logit = model.head(feat)
        c_target = concept_targets(ctable, y) if ctable is not None else None
        return (F.cross_entropy(logits, y)
                + concept_loss(c_logit, c_target, pos_weight=cweight)
                + model.head.part_loss(feat))
    if kind == "cbm":
        feat = model.features(x)
        logits, c_logit = model.head(feat)
        c_target = concept_targets(ctable, y) if ctable is not None else None
        return F.cross_entropy(logits, y) + concept_loss(c_logit, c_target,
                                                         pos_weight=cweight)
    return F.cross_entropy(model(x), y)


@torch.no_grad()
@torch.no_grad()
def _push_prototypes(model, loader, device, max_batches: int | None = None,
                     egg_only: bool = False, ann=None, img_size: int = 224,
                     margin: float = 0.20):
    """ProtoPNet projection: set each prototype to its nearest training patch embedding.

    Two correctness properties vs. the original:

    1. CLASS-RESTRICTED search (Chen et al. 2019). Prototype p belongs to class
       p // ppc (head.proto_class). It is projected ONLY onto patches from images of
       THAT class. Without this, a prototype can snap onto a patch of the wrong
       species — the "this looks like that" evidence would then display a patch from a
       different parasite than the class it votes for, which is unusable clinically.

    2. FULL-SET coverage by default. max_batches=None searches every training image
       (the paper's protocol). Pass an int only to deliberately subsample; record it if
       so. The push runs a handful of times over training, so full coverage is cheap.

    Any prototype whose class is absent from the scanned data keeps its current value
    (best_dist stays inf) — reported via the return value so a silent miss is visible.
    """
    model.eval()
    head = model.head
    P, D = head.num_protos, head.proto_dim
    ppc = head.ppc
    proto_of_class = [[p for p in range(P) if p // ppc == c]
                      for c in range(head.num_classes)]

    best_dist = torch.full((P,), float("inf"))
    best_vec = torch.zeros(P, D)

    # egg_only: iterate the dataset by index so file paths are available for the
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
            if getattr(head, "diverse_push", False) and len(protos_c) > 1:
                # assign the k prototypes to k DISTINCT patches (Hungarian) instead of
                # each taking its nearest independently, which lets them collapse
                from .models.protopnet_diverse import assign_diverse
                uniq = torch.unique(cand, dim=0)
                cols = assign_diverse(
                    head.prototypes[protos_c].view(len(protos_c), D),
                    uniq, len(protos_c))
                for j, p in enumerate(protos_c):
                    if j >= len(cols):
                        break
                    v = uniq[cols[j]]
                    md = ((v - head.prototypes[p].view(D)) ** 2).sum()
                    if md < best_dist[p]:
                        best_dist[p] = md.cpu()
                        best_vec[p] = v.detach().cpu()
            else:
                for p in protos_c:
                    pv = head.prototypes[p].view(1, D)
                    d = ((cand - pv) ** 2).sum(1)
                    md, mi = d.min(0)
                    if md < best_dist[p]:
                        best_dist[p] = md.cpu()
                        best_vec[p] = cand[mi].detach().cpu()

    missed = torch.isinf(best_dist)
    if missed.any():
        print(f"[push] WARNING: {int(missed.sum())}/{P} prototypes had no same-class "
              f"patch in the scanned data; kept previous value.", flush=True)
        # keep old values for missed prototypes
        best_vec[missed] = head.prototypes[missed].view(int(missed.sum()), D).detach().cpu()

    head.prototypes.copy_(best_vec.to(device).view(P, D, 1, 1))
    model.train()
    return {"pushed": int((~missed).sum()), "missed": int(missed.sum()),
            "egg_only": bool(egg_only), "fell_back": int(fell_back)}


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    train(load_config(args.config))