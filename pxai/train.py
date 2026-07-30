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
from .models.cbm import concept_loss


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
    best = 0.0
    out = ensure_dir(cfg["output_dir"])

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in tqdm(loaders.train, desc=f"ep{epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = _compute_loss(model, kind, x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        if kind == "protopnet" and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:
            # None = search the full training set (paper protocol). Set
            # train.push_max_batches in the config only to deliberately subsample.
            pmb = cfg["train"].get("push_max_batches", None)
            stats = _push_prototypes(model, loaders.train, device, max_batches=pmb)
            print(f"[push] epoch {epoch+1}: projected {stats['pushed']}/{model.head.num_protos} "
                  f"prototypes (coverage={'full' if pmb is None else f'{pmb} batches'})", flush=True)

        acc = evaluate_acc(model, loaders.val, device)
        print(f"epoch {epoch}: val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": loaders.classes,
                        "cfg": cfg}, f"{out}/best.pt")
    print(f"best val acc: {best:.4f}  -> {out}/best.pt")
    return model


def _compute_loss(model, kind, x, y):
    if kind == "protopnet":
        feat = model.features(x)
        logits = model.head(feat)
        ce = F.cross_entropy(logits, y)
        cluster, sep = model.head.cluster_sep_cost(feat, y)
        l1 = model.head.l1_last()
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1
    if kind == "cbm":
        feat = model.features(x)
        logits, c_logit = model.head(feat)
        # c_target wiring: plug morphology concept labels here when available
        return F.cross_entropy(logits, y) + concept_loss(c_logit, None)
    return F.cross_entropy(model(x), y)


@torch.no_grad()
@torch.no_grad()
def _push_prototypes(model, loader, device, max_batches: int | None = None):
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

    for bi, (x, y) in enumerate(loader):
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
                    best_vec[p] = patches[mi].detach().cpu()

    missed = torch.isinf(best_dist)
    if missed.any():
        print(f"[push] WARNING: {int(missed.sum())}/{P} prototypes had no same-class "
              f"patch in the scanned data; kept previous value.", flush=True)
        # keep old values for missed prototypes
        best_vec[missed] = head.prototypes[missed].view(int(missed.sum()), D).detach().cpu()

    head.prototypes.copy_(best_vec.to(device).view(P, D, 1, 1))
    model.train()
    return {"pushed": int((~missed).sum()), "missed": int(missed.sum())}


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