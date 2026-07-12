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
            _push_prototypes(model, loaders.train, device)

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
def _push_prototypes(model, loader, device, max_batches: int = 30):
    """ProtoPNet projection: set each prototype to its nearest training patch embedding."""
    model.eval()
    head = model.head
    best_dist = torch.full((head.num_protos,), float("inf"))
    best_vec = torch.zeros(head.num_protos, head.proto_dim)
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        z = head.add_on(model.features(x.to(device)))            # (B,D,H,W)
        B, D, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, D)                # (B*H*W, D)
        for p in range(head.num_protos):
            pv = head.prototypes[p].view(1, D)
            d = ((zf - pv) ** 2).sum(1)
            md, mi = d.min(0)
            if md < best_dist[p]:
                best_dist[p] = md.cpu()
                best_vec[p] = zf[mi].detach().cpu()
    with torch.no_grad():
        head.prototypes.copy_(best_vec.to(device).view(head.num_protos, head.proto_dim, 1, 1))
    model.train()


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
