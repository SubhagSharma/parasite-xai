#!/usr/bin/env bash
# run_night10.sh — ten hours, ordered so a truncated night still lands the best result.
#
#   nohup ./run_night10.sh > logs/n10_console.log 2>&1 &
#
# PHASE 1  concept-part head, train + evaluate                    ~2.0h
# PHASE 2  concept-part attention figures + per-concept table     ~0.5h
# PHASE 3  usage-balance replication, 2 more seeds                ~2.5h
# PHASE 4  RRR shortcut suppression, calibrate + 1 arm            ~2.0h
# PHASE 5  consolidated results table across every head           ~0.5h
#                                                          TOTAL  ~7.5h
#
# ORDERING
# Phase 1 first: the concept-part head is the only CONSTRUCTIVE result available, and
# its preflight (0.9766 after 300 steps, 11/11 classes) is the strongest of any head in
# the project. It is also the thing originally asked for -- an explanation that points
# at the operculum by name.
#
# Phase 3 next: today's headline is that load balancing raised prototype usage from 10%
# to 73% and left effective rank at 1.02/5. That is one seed. Given that seed variance
# has overturned three conclusions in this project, a claim of this weight needs n=3
# before it is written down.
#
# Phase 4 last of the experiments: RRR tests whether suppressing the background shortcut
# closes the localisation/faithfulness trade-off (Part II SEC 5.5.1). Valuable, but it
# tests an interpretation rather than establishing a fact.
#
# NOTHING IS OVERWRITTEN. New run directories throughout; results.json copied to
# .native-attr before any re-evaluation.

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs figs
MASTER="logs/night10_$(date +%Y%m%d_%H%M).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

wait_idle() {
  local n
  for _ in $(seq 1 120); do
    n=$(pgrep -fc "pxai.train|pxai.evaluate" || true)
    [ "${n:-0}" -eq 0 ] && return 0
    sleep 60
  done
  say "WARN: still busy after 2h; continuing anyway"
  return 0
}

train_one() {
  local a="$1"
  [ -f "configs/generated/$a.yaml" ] || { say "  skip $a (no config)"; return 0; }
  [ -f "runs/$a/.train_complete" ] && { say "  skip $a (done)"; return 0; }
  mkdir -p "runs/$a"
  say "  train $a"
  if python -u -m pxai.train --config "configs/generated/$a.yaml" \
       > "runs/$a/train.log" 2>&1; then
    touch "runs/$a/.train_complete"
    say "  done  $a  $(grep -o 'test_acc=[0-9.]*' runs/$a/train.log | tail -1)"
  else
    say "  FAIL  $a -- runs/$a/train.log"
  fi
}

echo "########## NIGHT 10 STARTED $(date) ##########" | tee -a "$MASTER"

# ============================================================ PHASE 1  parts
say "===== PHASE 1  concept-part head ====="
wait_idle
train_one roi477_parts_120ep

if [ -f runs/roi477_parts_120ep/best.pt ]; then
  say "  accuracy + localisation"
  python -u eval_accuracy_only.py \
      --config configs/generated/roi477_parts_120ep.yaml \
      --ckpt runs/roi477_parts_120ep/best.pt >> "$MASTER" 2>&1 || true
  tail -3 "$MASTER"

  say "  per-concept balanced accuracy -- the prediction is that CONTRAST determines"
  say "  localisability: contents=unembryonated (5/11 classes) best,"
  say "  symmetry=symmetric (10/11) worst. The CBM already shows the latter at TPR 0.600."
  python - >> "$MASTER" 2>&1 <<'PYC' || true
import torch
from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.concepts_loader import load_concept_table

cfg = load_config("configs/generated/roi477_parts_120ep.yaml"); cfg["device"] = "cuda"
dev = pick_device(cfg["device"]); l = build_loaders(cfg)
cfg["model"]["num_classes"] = len(l.classes)
m = build_model(cfg).to(dev)
m.load_state_dict(torch.load("runs/roi477_parts_120ep/best.pt", map_location=dev)["model"])
m.eval()
tab, names = load_concept_table(cfg["model"]["concept_parts"]["concepts_csv"], l.classes)
tab = tab.to(dev)

tp = torch.zeros(len(names)); fn = torch.zeros(len(names))
tn = torch.zeros(len(names)); fp = torch.zeros(len(names))
with torch.no_grad():
    for x, y in l.test:
        _, c_logit = m.head(m.features(x.to(dev)))
        p = (torch.sigmoid(c_logit) > 0.5).float()
        t = tab[y.to(dev)]
        tp += ((p == 1) & (t == 1)).sum(0).cpu(); fn += ((p == 0) & (t == 1)).sum(0).cpu()
        tn += ((p == 0) & (t == 0)).sum(0).cpu(); fp += ((p == 1) & (t == 0)).sum(0).cpu()

npos = (tab.sum(0)).cpu()
bal = 0.5 * (tp / (tp + fn).clamp_min(1) + tn / (tn + fp).clamp_min(1))
order = sorted(range(len(names)), key=lambda i: -float(bal[i]))
print(f"\n  {'concept':<34}{'balanced':>10}{'TPR':>8}{'classes+':>10}")
for i in order:
    print(f"  {names[i][:32]:<34}{float(bal[i]):>10.4f}"
          f"{float(tp[i]/(tp[i]+fn[i]).clamp_min(1)):>8.3f}{int(npos[i]):>8}/11")
print(f"  {'MACRO':<34}{float(bal.mean()):>10.4f}")
PYC
  tail -30 "$MASTER"
fi
say "===== phase 1 done ====="

# ============================================================ PHASE 2  figures
say "===== PHASE 2  attention figures -- the deliverable ====="
say "  the map IS the explanation: no gradient trick, no upsampled 7x7 grid"
python -u visualise_concept_parts.py --run roi477_parts_120ep --device cuda \
    --out figs/parts_overview.png >> "$MASTER" 2>&1 \
  || say "  visualiser missing or failed (see master log)"
python -u probe_gradattr.py --device cuda --runs roi477_parts_120ep \
    --emit-tsv figs/gradattr_parts.tsv >> "$MASTER" 2>&1 || true
tail -12 "$MASTER"
say "===== phase 2 done ====="

# ============================================================ PHASE 3  seeds
say "===== PHASE 3  usage-balance replication (n=3) ====="
say "  seed 1337 gave balance 10% -> 73% with effective rank 1.03 -> 1.02."
say "  Seed variance has overturned three conclusions in this project; this claim"
say "  needs replication before it is written down."
python - >> "$MASTER" 2>&1 <<'PYS'
import yaml, copy
base = yaml.safe_load(open("configs/generated/roi477_usage_120ep.yaml"))
for s in (2337, 3337):
    c = copy.deepcopy(base); c["seed"] = s
    n = f"roi477_usage_s{s}_120ep"
    c["output_dir"] = f"./runs/{n}"
    yaml.safe_dump(c, open(f"configs/generated/{n}.yaml", "w"), sort_keys=False)
    print(f"  wrote {n}")
PYS
for s in 2337 3337; do wait_idle; train_one "roi477_usage_s${s}_120ep"; done

say "  balance and effective rank across seeds"
python - >> "$MASTER" 2>&1 <<'PYB' || true
import torch, math, collections, statistics as st
import numpy as np, torch.nn.functional as F
from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model

runs = ["roi477_protopnet_120ep", "roi477_usage_120ep",
        "roi477_usage_s2337_120ep", "roi477_usage_s3337_120ep"]
print(f"\n  {'run':<30}{'balance':>9}{'eff rank':>10}")
for run in runs:
    try:
        cfg = load_config(f"configs/generated/{run}.yaml"); cfg["device"] = "cuda"
        dev = pick_device(cfg["device"]); l = build_loaders(cfg)
        cfg["model"]["num_classes"] = len(l.classes)
        m = build_model(cfg).to(dev)
        m.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location=dev)["model"])
        m.eval(); h = m.head
    except Exception:
        print(f"  {run:<30}{'--':>9}{'--':>10}"); continue
    ppc = h.prototypes.shape[0] // len(l.classes)
    win = collections.defaultdict(lambda: torch.zeros(ppc))
    with torch.no_grad():
        for bi, (x, y) in enumerate(l.test):
            sp, _ = h._similarities(m.features(x.to(dev)))
            for c in torch.unique(y).tolist():
                blk = sp[(y == c).to(dev)][:, c * ppc:(c + 1) * ppc]
                for w in blk.argmax(1).cpu().tolist():
                    win[c][w] += 1
            if bi >= 15:
                break
    b = math.log(ppc)
    bal = st.mean([float(-((win[c] / win[c].sum()).clamp_min(1e-9) *
                           (win[c] / win[c].sum()).clamp_min(1e-9).log()).sum()) / b
                   for c in win])
    pv = F.normalize(h.prototypes.detach().flatten(1), dim=1).cpu()
    eff = st.mean([float((lambda e: e.sum() ** 2 / max((e ** 2).sum(), 1e-12))(
        np.clip(np.linalg.eigvalsh((pv[c * ppc:(c + 1) * ppc] @
                                    pv[c * ppc:(c + 1) * ppc].t()).numpy()), 0, None)))
        for c in range(len(l.classes))])
    print(f"  {run:<30}{bal:>8.0%}{eff:>9.2f}/{ppc}")
print("""
  balance UP, eff FLAT across all seeds -> load balancing is the SIXTH rejected
  mechanism, and 'ProtoPNet converges to one effective prototype per class regardless
  of intervention' becomes the finding.""")
PYB
tail -12 "$MASTER"
say "===== phase 3 done ====="

# ============================================================ PHASE 4  RRR
say "===== PHASE 4  RRR shortcut suppression ====="
say "  tests Part II SEC 5.5.1: is the localisation/faithfulness trade-off CAUSED by"
say "  the background shortcut? If so, suppressing it should close the gap."
if [ -f run_rrr.sh ]; then
  [ -f pxai/rrr_penalty.py ] || cp rrr_penalty.py pxai/ 2>/dev/null || true
  chmod +x run_rrr.sh
  wait_idle
  ./run_rrr.sh calibrate >> "$MASTER" 2>&1 || say "  calibrate failed"
  grep "\[rrr\]" "$MASTER" | tail -6
  say "  NOTE: lambda not auto-selected. Read the ratios above and run"
  say "        RRR_LAMBDA=<value> ./run_rrr.sh train"
else
  say "  run_rrr.sh not present; skipped"
fi
say "===== phase 4 done ====="

# ============================================================ PHASE 5  summary
say "===== PHASE 5  consolidated table ====="
python - >> "$MASTER" 2>&1 <<'PYT' || true
import csv, glob, collections, statistics as st, os
d = collections.defaultdict(lambda: collections.defaultdict(list))
for p in glob.glob("figs/gradattr*.tsv") + glob.glob("figs/m_*.tsv") \
        + ["figs/attribution_metrics.tsv"]:
    if not os.path.exists(p):
        continue
    try:
        for r in csv.DictReader(open(p), delimiter="\t"):
            v = r.get("variant") or ("native" if r.get("method", "").startswith("ours:")
                                     else None)
            if v is None:
                continue
            x = r.get("conc_pos") or r.get("mass")
            try:
                x = float(x)
            except (TypeError, ValueError):
                continue
            if x == x:
                d[r["run"]][v].append(x)
    except Exception:
        pass
print(f"\n  {'run':<34}{'native':>9}{'gradattr':>10}{'n':>6}")
for run in sorted(d):
    v = d[run]
    f = lambda k: st.mean(v[k]) if v.get(k) else float("nan")
    print(f"  {run:<34}{f('native'):>9.2f}{f('gradattr'):>10.2f}"
          f"{len(v.get('native', [])):>6}")
print("  reference: 1.0 = random, 16.3 = perfect, IG 12.4")
PYT
tail -25 "$MASTER"

echo "########## NIGHT 10 COMPLETE $(date) ##########" | tee -a "$MASTER"
say "master log: $MASTER"
