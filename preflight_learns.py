"""
preflight_learns.py — does this config learn AT ALL? Two minutes, not five hours.

WHY
---
`A1_convnext_tiny_120ep` ran 120 epochs, 5.5 h of GPU, and sat at val_acc 0.0908-0.0914
from epoch 0 onward. No NaN, no exception, no warning. At inference it predicted a
single class for every input. Nothing in the pipeline noticed.

This runs ~300 optimiser steps and answers three questions:

  1. does the loss go down?
  2. is validation accuracy above chance?
  3. HOW MANY DISTINCT CLASSES does it predict?

(3) is the decisive one. The ConvNeXt failure was a collapse to one class, which
pins accuracy at exactly 1/num_classes and looks like "hasn't learned yet" from the
loss alone. A model predicting 1-2 distinct classes after 300 steps will still be
doing it at epoch 120.

It also prints `backbone.out_channels`, because the working hypothesis for that
collapse is sigmoid saturation in ProtoHead.add_on at 768 input channels versus
MobileViT-XS's 384.

CAVEAT: uses plain cross-entropy. ProtoPNet's cluster/separation terms and CBM's
concept term are not applied, so this is a smoke test for "can gradients move this
model", not a faithful reproduction of the training objective.

    python -u preflight_learns.py --config configs/generated/roi477_convnext_120ep.yaml
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["device"] = a.device
    dev = pick_device(cfg["device"])
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    loaders = build_loaders(cfg)
    ncls = len(loaders.classes)
    cfg["model"]["num_classes"] = ncls
    model = build_model(cfg).to(dev)

    ch = getattr(getattr(model, "backbone", None), "out_channels", "?")
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{cfg['backbone']['name']} + {cfg['model']['kind']}   "
          f"backbone.out_channels={ch}   {nparam:.2f}M params   "
          f"{ncls} classes, chance {1/ncls:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    model.train()

    losses, step = [], 0
    while step < a.steps:
        for x, y in loaders.train:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda",
                                dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"  step {step:>4}  loss {sum(losses[-50:]) / 50:.4f}", flush=True)
            if step >= a.steps:
                break

    first, last = sum(losses[:25]) / 25, sum(losses[-25:]) / 25

    model.eval()
    correct = total = 0
    seen = torch.zeros(ncls, dtype=torch.long)
    with torch.no_grad():
        for i, (x, y) in enumerate(loaders.val):
            if i >= a.eval_batches:
                break
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda",
                                dtype=torch.float16):
                p = model(x.to(dev)).float().argmax(1).cpu()
            seen += torch.bincount(p, minlength=ncls)
            correct += (p == y).sum().item()
            total += y.numel()
    acc = correct / max(total, 1)
    distinct = int((seen > 0).sum())

    print(f"\n  loss {first:.4f} -> {last:.4f}  ({(1 - last / first) * 100:+.1f}%)")
    print(f"  val accuracy after {a.steps} steps: {acc:.4f}  "
          f"({acc * ncls:.2f}x chance, n={total})")
    print(f"  distinct classes predicted: {distinct}/{ncls}")
    print(f"  prediction histogram: {seen.tolist()}")

    collapsed = distinct <= 2
    flat = last >= first * 0.95
    weak = acc < 1.5 / ncls
    print()
    if collapsed:
        print("  FAIL — COLLAPSED to a single class. This is the A1_convnext_tiny")
        print("  signature. It will not recover in 120 epochs. Do not queue it.")
    elif flat and weak:
        print("  FAIL — loss flat and accuracy at chance. Not learning.")
    elif weak:
        print("  WARN — above chance but weak. Could be a slow start; check again")
        print("  at epoch 5 rather than letting it run unattended for 120.")
    else:
        print("  PASS — loss falling, accuracy above chance, predictions spread")
        print("  across classes. Safe to queue.")
    raise SystemExit(2 if (collapsed or (flat and weak)) else 0)


if __name__ == "__main__":
    main()
