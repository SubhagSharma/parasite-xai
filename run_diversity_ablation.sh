#!/usr/bin/env bash
# run_diversity_ablation.sh — do the prototypes specialise onto distinct structures?
#
#   ./run_diversity_ablation.sh            # everything  (~2.5 h)
#   ./run_diversity_ablation.sh baseline   # measure the problem first, no training
#   ./run_diversity_ablation.sh train      # the 5 arms
#   ./run_diversity_ablation.sh measure    # diversity + localisation on all arms
#
# PHASE 0  baseline diversity on existing checkpoints   ~10m   no GPU training
# PHASE 1  train 5 arms, -P 3                           ~1.8h  ~85% GPU
# PHASE 2  diversity + localisation on all 5, -P 3      ~25m   ~85% GPU
#
# GPU: -P 3 keeps three MobileViT trainings resident at ~4 GB each. Single-process
# these models sit near 20% because they are launch-latency bound, not compute bound.
#
# THE TWO NUMBERS THAT MATTER, TOGETHER
#   eff       effective prototypes per class. Baseline is expected near 1.0 out of 5,
#             i.e. the class really has one prototype. Should RISE.
#   conc_pos  localisation. Should NOT fall.
#
# The egg covers 1-2 cells of a 7x7 grid, so diversity pressure that pushes prototypes
# onto different CELLS would put most of them on background. A rise in `cells` together
# with a fall in `conc_pos` means the pressure is too strong -- lower w_orth and rerun.
# Reporting only the diversity gain, with localisation hidden, would be the easy mistake.

set -euo pipefail
cd "$(dirname "$0")"

ARMS=(roi477_div_base_120ep roi477_div_orth_120ep roi477_div_orth_sp_120ep
      roi477_div_focal_120ep roi477_div_full_120ep)
BASE=roi477_protopnet_120ep
PAR=3
MASTER="logs/diversity_$(date +%Y%m%d_%H%M).log"

mkdir -p logs figs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

guard_idle() {
  local n; n=$(pgrep -fc "pxai.(train|evaluate)" || true)
  if [ "${n:-0}" -gt 0 ]; then
    say "ABORT: $n pxai process(es) running. Wait or kill them."
    pgrep -af "pxai.(train|evaluate)" | tee -a "$MASTER" || true
    exit 1
  fi
}

stage_baseline() {
  say "=== phase 0: how redundant are the CURRENT prototypes? ==="
  python -u probe_prototype_diversity.py --device cuda \
      --runs "$BASE,roi477_protopnet_s2337_120ep" \
      --emit-tsv figs/prototype_diversity.tsv 2>&1 | tail -25 | tee -a "$MASTER"
  say "  expected: eff near 1.0/5, cos near 1.0, map_corr > 0.8"
}

stage_train() {
  guard_idle
  say "=== phase 1: train ${#ARMS[@]} arms (-P $PAR) ==="
  for a in "${ARMS[@]}"; do
    [ -f "configs/generated/$a.yaml" ] || {
      say "ABORT: configs/generated/$a.yaml missing. Run make_diverse_configs.py."; exit 1; }
  done
  ./snapshot_stable.sh pre-diversity >>"$MASTER" 2>&1 || say "WARN: snapshot failed"

  printf '%s\n' "${ARMS[@]}" | xargs -P "$PAR" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/.train_complete" ] && { echo "  skip  $a"; exit 0; }
    mkdir -p "runs/$a"; echo "  start $a"
    if python -u -m pxai.train --config "configs/generated/$a.yaml" \
         > "runs/$a/train.log" 2>&1; then
      touch "runs/$a/.train_complete"
      echo "  done  $a  $(grep -o "test_acc=[0-9.]*" runs/$a/train.log | tail -1)"
    else
      echo "  FAIL  $a -- see runs/$a/train.log"
    fi' 2>&1 | tee -a "$MASTER"
  say "=== phase 1 finished ==="
}

stage_measure() {
  guard_idle
  say "=== phase 2: diversity + localisation (-P $PAR) ==="

  # diversity: one TSV, run serially (concurrent appends interleave and corrupt lines)
  for a in "${ARMS[@]}"; do
    [ -f "runs/$a/best.pt" ] || continue
    python -u probe_prototype_diversity.py --device cuda --runs "$a" \
        --emit-tsv figs/prototype_diversity.tsv > "logs/div_$a.log" 2>&1 \
      && echo "  div  $a" || echo "  FAIL div $a"
  done 2>&1 | tee -a "$MASTER"

  # localisation: separate TSV per arm, so this one can parallelise
  printf '%s\n' "${ARMS[@]}" | xargs -P "$PAR" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || exit 0
    python -u probe_gradattr.py --device cuda --runs "$a" \
        --emit-tsv "figs/gradattr_$a.tsv" > "logs/loc_$a.log" 2>&1 \
      && echo "  loc  $a" || echo "  FAIL loc $a"' 2>&1 | tee -a "$MASTER"

  say "=== the joint table: diversity gain must not cost localisation ==="
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, glob, collections, statistics as st, os

div = collections.defaultdict(lambda: collections.defaultdict(list))
if os.path.exists("figs/prototype_diversity.tsv"):
    for r in csv.DictReader(open("figs/prototype_diversity.tsv"), delimiter="\t"):
        for k in ("cos", "eff", "jacc", "map_corr", "cells"):
            try:
                div[r["run"]][k].append(float(r[k]))
            except (ValueError, KeyError):
                pass

loc = collections.defaultdict(list)
for p in glob.glob("figs/gradattr_*.tsv") + glob.glob("figs/m_*.tsv") \
        + ["figs/attribution_metrics.tsv"]:
    if not os.path.exists(p):
        continue
    try:
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r.get("variant") not in ("native", None) and "variant" in r:
                continue
            if r.get("method") not in (None, "ours:protopnet") and "method" in r:
                continue
            v = float(r.get("conc_pos", r.get("mass", "nan")))
            if v == v:
                loc[r["run"]].append(v)
    except Exception:
        pass

order = ["roi477_protopnet_120ep", "roi477_div_base_120ep", "roi477_div_orth_120ep",
         "roi477_div_orth_sp_120ep", "roi477_div_focal_120ep", "roi477_div_full_120ep"]
print(f"\n{'arm':<30}{'cos':>7}{'eff':>7}{'jacc':>7}{'map_corr':>10}"
      f"{'cells':>7}{'conc_pos':>10}")
print("-" * 78)
for run in order:
    if run not in div and run not in loc:
        continue
    d = div.get(run, {})
    def m(k):
        return st.mean(d[k]) if d.get(k) else float("nan")
    lc = st.mean(loc[run]) if loc.get(run) else float("nan")
    print(f"{run:<30}{m('cos'):>7.3f}{m('eff'):>7.2f}{m('jacc'):>7.3f}"
          f"{m('map_corr'):>10.3f}{m('cells'):>7.1f}{lc:>10.2f}")
print("""
  WANT   cos, jacc, map_corr DOWN   |   eff UP   |   conc_pos NOT DOWN
  WATCH  cells rising while conc_pos falls -> the diversity pressure is forcing
         prototypes off the egg onto background. Lower w_orth and rerun.
  CHECK  'div_base' must match 'protopnet' -- it is the control for the wiring.""")
PY
  say "=== phase 2 finished ==="
}

case "${1:-all}" in
  baseline) stage_baseline ;;
  train)    stage_train ;;
  measure)  stage_measure ;;
  all)      stage_baseline; stage_train; stage_measure ;;
  *)        echo "usage: $0 [all|baseline|train|measure]"; exit 2 ;;
esac

say "done. master log: $MASTER"
