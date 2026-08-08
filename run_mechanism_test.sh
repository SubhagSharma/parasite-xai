#!/bin/bash
# run_mechanism_test.sh — the two experiments that decide how §6 and §7.4 are written.
#
# PHASE 1  top-k localisation, 4 ROI heads        ~15m   resolves §7.4
# PHASE 2  448px config + preflight               ~5m
# PHASE 3  train roi477_protopnet_448             ~4h    tests the §6.4 mechanism
# PHASE 4  localisation + top-k on the 448 model  ~15m
#                                          TOTAL  ~4.6h
#
# WHY THESE TWO
# §7.4 records a contradiction: ProtoPNet has the best deletion score of 22 (0.0038)
# and the worst localisation (conc_pos 1.04, peak 24%). Phase 1 tests whether these
# measure different properties -- ranking vs mass -- in which case both readings are
# correct and the criticism in §6 must be softened from "does not find the egg" to
# "cannot draw a tight boundary around it".
#
# §6.4 attributes the localisation failure to spatial resolution: MobileViT-XS has
# stride 32, so a 224px input gives a 7x7 similarity map and one cell is 2.0% of the
# frame. Capillaria and Opisthorchis boxes are 2-4% -- one or two cells. At 448px the
# grid becomes 14x14 and one cell is 0.5%. Same architecture, same code, one config
# line. Phase 3 is therefore a direct test of the stated mechanism, not a tuning run.
#
#   conc_pos rises above ~4  -> the negative result becomes a DIAGNOSIS PLUS A FIX
#   conc_pos stays near 1.5  -> the cause is the prototype mechanism, not resolution
#
# Both outcomes are publishable; they lead to different papers.
#
#   chmod +x run_mechanism_test.sh
#   nohup ./run_mechanism_test.sh > runs/mechanism.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs figs
banner () { echo ""; echo "===== $1 :: $(date) ====="; }

ROI=../Data/chula_roi2_w477
BASE=roi477_protopnet_120ep
NEW=roi477_protopnet_448_120ep

echo "########## MECHANISM TEST STARTED $(date) ##########"

banner "PHASE 1  top-k localisation — does the explanation RANK the egg?"
# One process per run, 4 in parallel. These models are launch-latency bound at batch 1
# (~20% utilisation single-process); 4 concurrent CUDA contexts at ~2 GB each fill the
# slice and cover each other's gaps. Separate TSVs because concurrent appends to one
# file interleave and corrupt lines.
rm -f figs/topk_p_*.tsv
echo "roi477_protopnet_120ep
roi477_bcos_120ep
roi477_cbm_sup_120ep
roi477_blackbox_120ep" | xargs -P 4 -I{} sh -c \
  'python -u probe_topk_localisation.py --runs "{}" --device cuda \
     --emit-tsv figs/topk_p_{}.tsv > runs/topk_{}.log 2>&1'

head -1 $(ls figs/topk_p_*.tsv | head -1) > figs/topk_localisation.tsv
tail -q -n +2 figs/topk_p_*.tsv >> figs/topk_localisation.tsv
rm -f figs/topk_p_*.tsv
echo "merged $(($(wc -l < figs/topk_localisation.tsv) - 1)) rows"

python - <<'PY2'
import csv, collections, statistics as st
KS=["top0.1pct","top1pct","top5pct","top10pct"]
rows=list(csv.DictReader(open('figs/topk_localisation.tsv'), delimiter='\t'))
d=collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try: a=float(r['area'])
    except (ValueError,KeyError): continue
    if a<=0: continue
    k=(r['head'],r['method'])
    for c in KS:
        try: d[k][c].append(float(r[c])/a)
        except (ValueError,KeyError): pass
    for c in ('best_rank_pct','mass_conc'):
        try:
            v=float(r[c])
            if v==v: d[k][c].append(v)
        except (ValueError,KeyError): pass
print(f"\n{'head':<11}{'method':<22}{'c@0.1%':>9}{'c@1%':>8}{'c@5%':>8}{'c@10%':>8}"
      f"{'mass':>8}{'1st hit':>9}{'n':>6}")
print("-"*90)
for k in sorted(d, key=lambda k: -st.mean(d[k]['top1pct'] or [0])):
    v=d[k]
    if not v['top1pct']: continue
    print(f"{k[0]:<11}{k[1]:<22}"+"".join(f"{st.mean(v[c]):>8.2f} " for c in KS)
          +f"{st.mean(v['mass_conc']):>7.2f}{st.mean(v['best_rank_pct']):>8.1f}%"
          f"{len(v['top1pct']):>6}")
print("""
  c@1% >> mass -> SEC 7.4 DISSOLVES: ranks the egg, spreads its mass. Both readings
                  correct; soften SEC 6 to "cannot draw a tight boundary".
  c@1% ~= mass -> SEC 7.4 REAL: neither ranks nor locates. Audit deletion.""")
PY2

banner "PHASE 2  448px config"
python - <<'PY'
import yaml, os
src = "configs/generated/roi477_protopnet_120ep.yaml"
dst = "configs/generated/roi477_protopnet_448_120ep.yaml"
c = yaml.safe_load(open(src))
c["data"]["img_size"] = 448          # stride 32 -> 14x14 grid, one cell = 0.51% of frame
# Keep batch_size 32, IDENTICAL to the 224 baseline. 4x the pixels at the same batch
# is ~4x the activation memory (~16 GB of the 20 GB slice, ~80% utilisation), and it
# keeps the comparison controlled: changing batch size would alter the effective
# learning rate and normalisation statistics alongside the resolution.
# If this OOMs, drop to 24 and record that the comparison is no longer batch-matched.
c["data"]["batch_size"] = 32
c["output_dir"] = "./runs/roi477_protopnet_448_120ep"
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print(f"wrote {dst}")
print("  img_size 224 -> 448; batch_size 32 unchanged; everything else unchanged")
PY

banner "PHASE 2b  preflight — does it learn at all?"
python -u preflight_learns.py --config configs/generated/$NEW.yaml --device cuda
PRE=$?
echo "preflight exit $PRE"
if [ $PRE -ne 0 ]; then
  echo "PREFLIGHT FAILED — not queueing a 4h run."
  echo "If the cause was CUDA OOM, retry with batch_size 24:"
  echo "  sed -i 's/batch_size: 32/batch_size: 24/' configs/generated/$NEW.yaml"
  echo "and note in the writeup that the 224/448 comparison is no longer batch-matched."
  exit 1
fi
nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null || true

banner "PHASE 3  train $NEW"
if [ -f "runs/$NEW/.train_complete" ]; then
  echo "SKIP — already complete"
else
  mkdir -p "runs/$NEW"
  python -u -m pxai.train --config "configs/generated/$NEW.yaml" \
      > "runs/$NEW/train.log" 2>&1
  c=$?
  [ $c -eq 0 ] && touch "runs/$NEW/.train_complete" || echo "TRAIN FAILED ($c)"
  echo "exit $c"
fi

banner "PHASE 4  localisation on the 448 model"
if [ -f "runs/$NEW/best.pt" ]; then
  python -u eval_accuracy_only.py --config "configs/generated/$NEW.yaml" \
      --ckpt "runs/$NEW/best.pt" 2>&1 | tee -a "runs/$NEW/accuracy.log"

  python -u batch_visualise.py --runs "$NEW" \
      --outdir figs --tsv figs/m_${NEW}.tsv --device cuda

  python -u probe_topk_localisation.py --runs "$NEW" \
      --emit-tsv figs/topk_localisation.tsv --device cuda

  banner "COMPARISON — 224px vs 448px, ProtoPNet own explanation"
  python - <<'PY'
import csv, collections, statistics as st, os
def load(p):
    return list(csv.DictReader(open(p), delimiter='\t')) if os.path.exists(p) else []
rows = load('figs/attribution_metrics.tsv') + load('figs/m_roi477_protopnet_448_120ep.tsv')
d = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r.get('method') != 'ours:protopnet':
        continue
    for k in ('conc_pos', 'peak', 'area'):
        try:
            v = float(r[k])
            if v == v:
                d[r['run']][k].append(v)
        except (ValueError, KeyError):
            pass
print(f"{'run':<34}{'conc_pos':>10}{'peak%':>8}{'box area':>10}{'n':>6}")
print("-" * 68)
for run in sorted(d):
    v = d[run]
    if not v['conc_pos']:
        continue
    print(f"{run:<34}{st.mean(v['conc_pos']):>10.2f}{st.mean(v['peak'])*100:>7.0f}%"
          f"{st.mean(v['area'])*100:>9.1f}%{len(v['conc_pos']):>6}")
print("""
224px -> 7x7 grid, one cell = 2.04% of frame
448px -> 14x14 grid, one cell = 0.51% of frame

If conc_pos rose substantially on the 448 model, the localisation failure is a
RESOLUTION limit and the fix is a finer feature grid. If it did not, the cause is
the prototype mechanism itself and the negative result stands unqualified.""")
PY
else
  echo "no checkpoint — phase 4 skipped"
fi

echo ""
echo "########## DONE $(date) ##########"