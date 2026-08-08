#!/usr/bin/env bash
# run_after_usage.sh — wait for the usage run, then measure it, then build and train
# the spatially-grounded concept-part head.
#
# Serial by design: both want the GPU, and the usage result decides whether the
# prototype line is worth continuing at all.
#
#   nohup ./run_after_usage.sh > logs/after_usage.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs runs/roi477_parts_120ep
say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- wait + measure
say "waiting for roi477_usage_120ep"
while pgrep -f "pxai.train.*roi477_usage" > /dev/null; do sleep 120; done
say "usage run finished"

say "=== usage result: did load balancing raise prototype diversity? ==="
python - <<'PY'
import torch, math, collections, itertools, statistics as st
import numpy as np, torch.nn.functional as F
from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model

for run in ("roi477_protopnet_120ep", "roi477_usage_120ep"):
    try:
        cfg = load_config(f"configs/generated/{run}.yaml"); cfg["device"] = "cuda"
        dev = pick_device(cfg["device"]); l = build_loaders(cfg)
        cfg["model"]["num_classes"] = len(l.classes)
        m = build_model(cfg).to(dev)
        m.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location=dev)["model"])
        m.eval(); h = m.head
    except Exception as e:
        print(f"{run}: {type(e).__name__}: {e}"); continue
    ppc = h.prototypes.shape[0] // len(l.classes)

    win = collections.defaultdict(lambda: torch.zeros(ppc))
    with torch.no_grad():
        for bi, (x, y) in enumerate(l.test):
            sp, _ = h._similarities(m.features(x.to(dev)))
            for c in torch.unique(y).tolist():
                blk = sp[(y == c).to(dev)][:, c*ppc:(c+1)*ppc]
                for w in blk.argmax(1).cpu().tolist():
                    win[c][w] += 1
            if bi >= 15: break
    b = math.log(ppc)
    bal = st.mean([float(-( (win[c]/win[c].sum()).clamp_min(1e-9) *
                            (win[c]/win[c].sum()).clamp_min(1e-9).log()).sum()) / b
                   for c in win])

    pv = F.normalize(h.prototypes.detach().flatten(1), dim=1).cpu()
    eff = st.mean([float((lambda ev: ev.sum()**2 / max((ev**2).sum(), 1e-12))(
        np.clip(np.linalg.eigvalsh((pv[c*ppc:(c+1)*ppc] @
                                    pv[c*ppc:(c+1)*ppc].t()).numpy()), 0, None)))
        for c in range(len(l.classes))])
    print(f"  {run:<28} balance {bal:>5.0%}   eff rank {eff:>4.2f}/{ppc}")
print("""
  baseline: balance 10%, eff 1.03/5
  balance UP and eff UP    -> winner-take-all was the cause; a one-term fix
  balance UP, eff FLAT     -> prototypes now share samples but not meaning;
                              redundancy is deeper than assignment""")
PY

# ---------------------------------------------------------------- concept parts
say "=== wiring the concept-part head ==="
[ -f pxai/models/concept_parts.py ] || cp concept_parts.py pxai/models/
grep -q ConceptPartHead pxai/models/__init__.py 2>/dev/null || {
  python apply_concept_parts.py --check && python apply_concept_parts.py; }
python -c "import pxai.train, pxai.models; print('  imports OK')" || exit 1
python make_parts_config.py

say "=== preflight ==="
python -u preflight_learns.py --config configs/generated/roi477_parts_120ep.yaml \
    --device cuda 2>&1 | tail -6

say "=== train (~1.5h) ==="
python -u -m pxai.train --config configs/generated/roi477_parts_120ep.yaml \
    > runs/roi477_parts_120ep/train.log 2>&1 \
  && touch runs/roi477_parts_120ep/.train_complete || say "TRAIN FAILED"

say "=== do the attention maps land on the egg? ==="
python -u probe_gradattr.py --device cuda --runs roi477_parts_120ep \
    --emit-tsv figs/gradattr_parts.tsv 2>&1 | tail -8 || true

say "done"
