"""preflight_sea_localises.py — does the SEA head localise a KNOWN signal?

`preflight_learns.py` catches a model that never learns. This catches the other
failure: a model that classifies perfectly while its attribution map stays
uniform -- exactly the ProtoPNet outcome (conc 1.04, peak-in-box 24%). Run it
before queueing any SEA config overnight.

    python preflight_sea_localises.py --device cuda
    python preflight_sea_localises.py --device cuda --stride 4 --steps 400

Synthetic task: a coloured disc at a RANDOM position on textured noise; the
class is the disc's colour, never its location, so position carries no label
information and a model that scores well on `conc` can only have done it by
finding the disc. Ground truth is the disc's box, and `conc` / `peak` are
computed with the same definitions as batch_visualise.py, so the numbers are
directly comparable to the ones in the report.

Pass condition: conc clearly above 1.0 and peak-in-box well above the area
baseline, WITH the concentration prior on and near-baseline with it off. If
both arms sit at 1.0 the head is not localising and no amount of GPU time on
the real dataset will fix it.

Runtime: ~90 s on CPU at 64 px, ~40 s on GPU at 224 px.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from pxai.models.sea import SEANet, sea_loss

K = 4                                   # synthetic classes


def make_batch(n, size, radius, device, rng):
    """-> x (n,3,S,S), y (n,), boxes (n,4) as x0,y0,x1,y1."""
    lo = torch.from_numpy(rng.normal(0, 1, (n, 3, size // 8, size // 8))).float()
    x = F.interpolate(lo, size=(size, size), mode="bilinear",
                      align_corners=False) * 0.5
    y = torch.from_numpy(rng.integers(0, K, n)).long()
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    boxes = torch.zeros(n, 4, dtype=torch.long)
    palette = torch.tensor([[1.5, -1.0, -1.0], [-1.0, 1.5, -1.0],
                            [-1.0, -1.0, 1.5], [1.2, 1.2, -1.2]])
    for i in range(n):
        cx = int(rng.integers(radius + 1, size - radius - 1))
        cy = int(rng.integers(radius + 1, size - radius - 1))
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
        x[i, :, disc] = palette[y[i]].view(3, 1)
        boxes[i] = torch.tensor([cx - radius, cy - radius,
                                 cx + radius, cy + radius])
    return x.to(device), y.to(device), boxes


def loc_metrics(attr, boxes):
    """frac, area, conc, peak -- same definitions as batch_visualise.metrics."""
    a = attr.abs().double()
    B, H, W = a.shape
    out = []
    for i in range(B):
        m = torch.zeros(H, W, dtype=torch.bool)
        x0, y0, x1, y1 = boxes[i].tolist()
        m[y0:y1 + 1, x0:x1 + 1] = True
        tot = a[i].sum()
        if tot <= 0:
            continue
        frac = float(a[i][m].sum() / tot)
        area = float(m.double().mean())
        pk = divmod(int(a[i].argmax()), W)
        out.append((frac, area, frac / area, float(m[pk[0], pk[1]])))
    f, ar, c, p = map(lambda v: float(np.mean(v)), zip(*out))
    return f, ar, c, p


def run_arm(label, conc, args, device):
    rng = np.random.default_rng(args.seed)
    cfg = {"backbone": {"name": args.backbone, "pretrained": False},
           "model": {"kind": "sea", "num_classes": K,
                     "sea": {"stride": args.stride, "dim": 48, "depth": 2,
                             "readout": "mlp", "context": "none",
                             "max_evidence_stride": 8}}}
    torch.manual_seed(args.seed)
    m = SEANet(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    w = {"conc": conc, "tv": 0.1 if conc > 0 else 0.0, "tau": 0.15}

    m.train()
    for step in range(args.steps):
        x, y, _ = make_batch(args.batch, args.size, args.radius, device, rng)
        loss, st = sea_loss(m, x, y, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % max(1, args.steps // 4) == 0:
            print(f"     step {step+1:>4}  loss={loss.item():6.3f}  "
                  f"ce={st['ce']:6.3f}  pr_frac={st.get('pr_frac', float('nan')):.3f}")

    m.eval()
    x, y, boxes = make_batch(256, args.size, args.radius, device,
                             np.random.default_rng(args.seed + 999))
    with torch.no_grad():
        ev = m.explain(x)["contrib_map"]
        acc = float((m(x).argmax(1) == y).float().mean())
        attr = ev.gather(1, y.view(-1, 1, 1, 1).expand(-1, 1, *ev.shape[-2:]))
        attr = F.interpolate(attr, size=(args.size, args.size), mode="bilinear",
                             align_corners=False)[:, 0].cpu()
    f, ar, c, p = loc_metrics(attr, boxes)
    print(f"  {label:<22} acc={acc:.3f}  conc={c:.2f}  peak-in-box={p*100:.0f}%"
          f"  (box area {ar*100:.1f}%, uniform conc = 1.00, "
          f"chance peak = {ar*100:.0f}%)")
    return acc, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    dev = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available()
                       else "cpu")
    print(f"device={dev}  backbone={a.backbone}  size={a.size}  "
          f"stride={a.stride}  grid={a.size // a.stride}x{a.size // a.stride}")

    print("\n  arm A: concentration prior OFF")
    acc0, c0, p0 = run_arm("prior off", 0.0, a, dev)
    print("\n  arm B: concentration prior ON")
    acc1, c1, p1 = run_arm("prior on", 1.0, a, dev)

    print("\n  ---")
    ok = acc1 > 0.9 and c1 > 1.5 and c1 > c0
    print(f"  accuracy cost of the prior: {acc0 - acc1:+.3f}")
    print(f"  concentration gain:         {c0:.2f} -> {c1:.2f}")
    print(f"  peak-in-box gain:           {p0*100:.0f}% -> {p1*100:.0f}%")
    print("  " + ("PASS -- the head localises; queue the real run."
                  if ok else
                  "FAIL -- do not queue a 3 h run. Raise conc, lower tau, or "
                  "drop the stride first."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
