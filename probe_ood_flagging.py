r"""
probe_ood_flagging.py — IPI-CVx OOD sample selection, step 1.

IPI-CVx flags "high-risk" training samples where CONFIDENCE and CORRECTNESS
disagree, then sends them for expert review, augments them, and fine-tunes on them
(ASTL). Their two conditions, with delta = 0.85 and tau = 0.7:

    IoU > delta and MSP > tau and y_gt != y_pred  -> high-confidence FALSE POSITIVE
                                                     ("over-confidence")
    IoU > delta and MSP < tau and y_gt == y_pred  -> low-confidence TRUE POSITIVE
                                                     ("under-confidence")

NOTE ON IoU. Their IoU compares a DETECTOR's predicted box against ground truth.
This pipeline is a classifier, so there is no predicted box. On the CROPPED dataset
that is not a problem: the crop IS the annotated ROI, so IoU = 1.0 by construction
and `IoU > delta` is satisfied for every sample. The condition therefore reduces to
the MSP test alone -- which is exact here rather than an approximation.

This script only MEASURES and flags. It does not modify training data. The two
downstream options are opposite interventions and must be chosen deliberately:

    ADD BACK (IPI-CVx / ASTL): augment the flagged samples and fine-tune on them.
        Targets robustness to intra-class variation. This is what the paper does.
    REMOVE: drop the flagged samples from training. Measures how much accuracy
        depends on ambiguous cases. Will likely RAISE test accuracy while making
        the model worse on real variation. Legitimate, but it is NOT ASTL.

Writes runs/<name>/ood_flags.json with the flagged sample paths and categories.

    python probe_ood_flagging.py \
        --config configs/generated/crop_protopnet_120ep.yaml \
        --ckpt   runs/crop_protopnet_120ep/best.pt \
        --tau 0.7 --k 0.02
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model


def split_paths(subset_or_ds):
    ds = subset_or_ds
    if isinstance(ds, Subset):
        base = ds.dataset
        while isinstance(base, Subset):
            base = base.dataset
        return [base.samples[i][0] for i in ds.indices]
    return [p for p, _ in ds.samples]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tau", type=float, default=0.7, help="MSP threshold (paper: 0.7)")
    ap.add_argument("--k", type=float, default=0.02,
                    help="fraction of train set to keep after ranking (paper: 0.02)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device
    dev = pick_device(cfg["device"])

    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()

    paths = split_paths(loaders.train.dataset)
    loader = DataLoader(loaders.train.dataset, batch_size=cfg["data"]["batch_size"],
                        shuffle=False, num_workers=2)

    msp_all, pred_all, y_all = [], [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(dev)).float(), 1)
        m, pr = p.max(1)
        msp_all.append(m.cpu().numpy())
        pred_all.append(pr.cpu().numpy())
        y_all.append(y.numpy())
    msp = np.concatenate(msp_all)
    pred = np.concatenate(pred_all)
    y = np.concatenate(y_all)
    n = len(y)

    correct = pred == y
    high_conf_fp = (~correct) & (msp > args.tau)     # over-confidence
    low_conf_tp = correct & (msp < args.tau)         # under-confidence
    flagged = high_conf_fp | low_conf_tp

    print(f"\n=== OOD flagging on the TRAIN split ({n} crops) ===")
    print(f"  IoU > delta : satisfied by construction (the crop IS the ROI)")
    print(f"  tau (MSP)   : {args.tau}")
    print(f"\n  train accuracy            : {correct.mean():.4f}")
    print(f"  mean MSP                  : {msp.mean():.4f}")
    print(f"  high-confidence FP (over) : {int(high_conf_fp.sum()):>5}  "
          f"({high_conf_fp.mean()*100:.2f}%)")
    print(f"  low-confidence TP (under) : {int(low_conf_tp.sum()):>5}  "
          f"({low_conf_tp.mean()*100:.2f}%)")
    print(f"  TOTAL flagged             : {int(flagged.sum()):>5}  "
          f"({flagged.mean()*100:.2f}%)")
    print(f"  paper's budget k={args.k}     : {int(round(args.k*n)):>5} samples")

    # rank by how extreme the disagreement is, so k can be applied
    score = np.where(high_conf_fp, msp, np.where(low_conf_tp, 1.0 - msp, -1.0))
    order = np.argsort(-score)
    keep = [i for i in order if flagged[i]][: int(round(args.k * n))]

    print(f"\n{'class':>26} {'over':>6} {'under':>7} {'total':>7}")
    per = defaultdict(lambda: [0, 0])
    for i in range(n):
        if high_conf_fp[i]:
            per[loaders.classes[y[i]]][0] += 1
        elif low_conf_tp[i]:
            per[loaders.classes[y[i]]][1] += 1
    for c in loaders.classes:
        o, u = per[c]
        print(f"{c[:26]:>26} {o:>6} {u:>7} {o+u:>7}")

    out = os.path.join(cfg["output_dir"], "ood_flags.json")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "tau": args.tau, "k": args.k, "n_train": n,
            "train_accuracy": float(correct.mean()),
            "n_high_conf_fp": int(high_conf_fp.sum()),
            "n_low_conf_tp": int(low_conf_tp.sum()),
            "topk_indices": [int(i) for i in keep],
            "flagged": [
                {"path": paths[i],
                 "true": loaders.classes[int(y[i])],
                 "pred": loaders.classes[int(pred[i])],
                 "msp": float(msp[i]),
                 "type": "high_conf_fp" if high_conf_fp[i] else "low_conf_tp"}
                for i in keep],
        }, f, indent=2)
    print(f"\n  wrote {out}  ({len(keep)} samples at k={args.k})")
    print("\n  NEXT: decide ADD BACK (ASTL, the paper's protocol) or REMOVE")
    print("  (your variant). They test opposite hypotheses.\n")


if __name__ == "__main__":
    main()
