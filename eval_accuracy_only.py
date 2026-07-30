"""
eval_accuracy_only.py — test accuracy for a checkpoint, in ~1 minute.

The 2x2 head-vs-backbone control needs ACCURACY only, not the faithfulness sweep.
Running the full eval on those control models would cost hours per model for
metrics that answer a different question.

    python eval_accuracy_only.py --config configs/generated/<name>.yaml \
                                 --ckpt   runs/<name>/best.pt

Prints test accuracy, val accuracy, and cost (params / size), and appends a line to
runs/accuracy_2x2.tsv so the comparison table builds up as runs finish.
"""
from __future__ import annotations

import argparse
import os

import torch

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.train import evaluate_acc
from pxai.eval import cost as costmod

TSV = "runs/accuracy_2x2.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None, help="override cfg device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    device = pick_device(cfg["device"])

    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    test_acc = evaluate_acc(model, loaders.test, device)
    val_acc = evaluate_acc(model, loaders.val, device)
    params = costmod.count_params(model) / 1e6
    size = costmod.model_size_mb(model)

    name = os.path.basename(os.path.dirname(args.ckpt))
    backbone = cfg["backbone"]["name"]
    kind = cfg["model"]["kind"]

    print(f"\n{'run':>28} {'backbone':>16} {'head':>10} {'val':>8} {'test':>8} "
          f"{'params M':>9} {'size MB':>8}")
    print(f"{name:>28} {backbone:>16} {kind:>10} {val_acc:>8.4f} {test_acc:>8.4f} "
          f"{params:>9.3f} {size:>8.2f}\n")

    os.makedirs("runs", exist_ok=True)
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("run\tbackbone\thead\tval_acc\ttest_acc\tparams_M\tsize_MB\n")
        f.write(f"{name}\t{backbone}\t{kind}\t{val_acc:.4f}\t{test_acc:.4f}"
                f"\t{params:.3f}\t{size:.2f}\n")
    print(f"appended to {TSV}")


if __name__ == "__main__":
    main()
