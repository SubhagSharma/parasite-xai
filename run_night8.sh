#!/usr/bin/env bash
# run_night8.sh — close three cheap weaknesses, then answer the open Part II question.
#
#   ./run_night8.sh            # everything  (~7 h)
#   ./run_night8.sh audit      # what seeds actually exist       ~2m
#   ./run_night8.sh seeds      # fill any real seed gaps         ~1.5h
#   ./run_night8.sh bcos       # is B-cos just HiResCAM?         ~15m
#   ./run_night8.sh push       # is best.pt post-push?           ~20m
#   ./run_night8.sh faith      # faithfulness under gradattr     ~6h
#
# THE THREE CHEAP FIXES
# Each removes an objection an examiner would otherwise raise, and each costs under two
# hours. They were listed as caveats when they should have been fixed.
#
#   1. SEED COUNT. B-cos and CBM appeared to have n=3 against ProtoPNet's n=5 -- but
#      that count came from a glob over figs/m_*_n7.tsv, which only covers night 7.
#      Checkpoints from nights 5 and 6 may already close the gap. AUDIT FIRST; train
#      only what is genuinely missing.
#
#   2. B-COS vs HIRESCAM. They tie at deletion to four decimal places on two datasets
#      (0.2421 whole, 0.0465 roi477). If the maps correlate at ~1.0 this is not a
#      weakness but a RESULT: the B-cos-style head's explanation is mathematically a
#      gradient-weighted CAM, which questions whether it is a distinct interpretability
#      mechanism at all. Turns an attack into a finding.
#
#   3. PROTOTYPE PLACEMENT. best.pt is a BEST-VALIDATION checkpoint, and push runs
#      after training on every 10th epoch -- so the saved prototypes need not follow a
#      push. Chen et al.'s "this looks like that" requires each prototype to BE a real
#      patch. Re-push at eval time with EVAL transforms and measure the drift. Small
#      drift => the property holds. Large => it does not, and Part I says so.
#      (No retrain. The earlier 0/55 result was confounded by augmentation, which this
#      avoids by construction.)
#
# THEN THE OPEN QUESTION
# Part II recovered localisation but never re-measured faithfulness. Deletion,
# insertion and sensitivity are still computed on the native map. Either outcome is
# publishable and the ambiguity is not.
#
#   chmod +x run_night8.sh
#   nohup ./run_night8.sh > logs/night8_console.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")"

PAR_TRAIN=3
PAR_EVAL=2
MASTER="logs/night8_$(date +%Y%m%d_%H%M).log"
FAITH=(roi477_protopnet_120ep roi477_cbm_sup_120ep roi477_bcos_120ep)

mkdir -p logs figs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

guard_idle() {
  local n; n=$(pgrep -fc "pxai.(train|evaluate)" || true)
  if [ "${n:-0}" -gt 0 ]; then
    say "ABORT: $n pxai process(es) running."
    pgrep -af "pxai.(train|evaluate)" | tee -a "$MASTER" || true
    exit 1
  fi
}

# ------------------------------------------------------------------ stage: audit
stage_audit() {
  say "=== stage: audit — which seeds actually have checkpoints? ==="
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import glob, os, re, collections
have = collections.defaultdict(list)
for p in sorted(glob.glob("runs/roi477_*/best.pt")):
    r = p.split("/")[1]
    m = re.match(r"roi477_(blackbox|protopnet|bcos|cbm_sup|cbm)(?:_s(\d+))?_120ep$", r)
    if m and "div" not in r and "eggpush" not in r:
        have[m.group(1)].append(int(m.group(2) or 1337))
print(f"{'head':<12}{'n':>3}  seeds")
for h in ("protopnet", "bcos", "cbm_sup", "blackbox"):
    s = sorted(have.get(h, []))
    print(f"{h:<12}{len(s):>3}  {s}")
gaps = {h: sorted({1337,2337,3337,4337,5337} - set(have.get(h, [])))
        for h in ("protopnet", "bcos", "cbm_sup", "blackbox")}
missing = {h: g for h, g in gaps.items() if g}
note = missing if missing else "nothing -- the n=3 concern was a glob artefact"
print(f"\nmissing for n=5: {note}")
open("/tmp/seed_gaps.txt", "w").write(
    "\n".join(f"roi477_{h}_s{s}_120ep" for h, g in missing.items() for s in g
              if s != 1337))
PY
  say "  gaps written to /tmp/seed_gaps.txt"
}

# ------------------------------------------------------------------ stage: seeds
stage_seeds() {
  guard_idle
  say "=== stage: seeds — train only the genuine gaps ==="
  [ -s /tmp/seed_gaps.txt ] || { say "  no gaps; skipping"; return 0; }
  wc -l < /tmp/seed_gaps.txt | xargs -I{} say "  {} run(s) to train"

  python - <<'PY' 2>&1 | tee -a "$MASTER"
import yaml, copy, os, re
for name in [l.strip() for l in open("/tmp/seed_gaps.txt") if l.strip()]:
    if os.path.exists(f"configs/generated/{name}.yaml"):
        print(f"  {name}: config exists"); continue
    m = re.match(r"roi477_(\w+?)_s(\d+)_120ep", name)
    src = f"configs/generated/roi477_{m.group(1)}_120ep.yaml"
    if not os.path.exists(src):
        print(f"  {name}: SKIP, no source {src}"); continue
    c = yaml.safe_load(open(src)); c["seed"] = int(m.group(2))
    c["output_dir"] = f"./runs/{name}"
    yaml.safe_dump(c, open(f"configs/generated/{name}.yaml", "w"), sort_keys=False)
    print(f"  {name}: config written")
PY

  xargs -P "$PAR_TRAIN" -I{} bash -c '
    a="{}"
    [ -f "configs/generated/$a.yaml" ] || { echo "  skip  $a (no config)"; exit 0; }
    [ -f "runs/$a/.train_complete" ] && { echo "  skip  $a"; exit 0; }
    mkdir -p "runs/$a"; echo "  start $a"
    python -u -m pxai.train --config "configs/generated/$a.yaml" \
      > "runs/$a/train.log" 2>&1 && touch "runs/$a/.train_complete" \
      && echo "  done  $a" || echo "  FAIL  $a"' < /tmp/seed_gaps.txt 2>&1 | tee -a "$MASTER"

  say "  re-measuring localisation for EVERY head and seed into one TSV"
  rm -f figs/all_seeds.tsv
  for a in $(ls -d runs/roi477_*_120ep 2>/dev/null | xargs -n1 basename \
             | grep -vE "div|eggpush|448|effnet|ghost|convnext"); do
    [ -f "runs/$a/best.pt" ] || continue
    python -u probe_gradattr.py --device cuda --runs "$a" \
        --emit-tsv figs/all_seeds.tsv > "logs/gs_$a.log" 2>&1 \
      && echo "  meas $a" || echo "  FAIL meas $a"
  done 2>&1 | tee -a "$MASTER"

  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, collections, statistics as st, os
if not os.path.exists("figs/all_seeds.tsv"):
    raise SystemExit("  no TSV")
d = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open("figs/all_seeds.tsv"), delimiter="\t"):
    if r["variant"] != "native":
        continue
    try:
        d[r["head"]][r["run"]].append(float(r["mass"]))
    except ValueError:
        pass
print(f"\n{'head':<12}{'mean':>7}{'min':>7}{'max':>7}{'spread':>8}{'seeds':>7}")
for h, runs in sorted(d.items()):
    v = [st.mean(x) for x in runs.values()]
    print(f"{h:<12}{st.mean(v):>7.2f}{min(v):>7.2f}{max(v):>7.2f}"
          f"{max(v)/max(min(v),1e-9):>7.2f}x{len(v):>7}")
PY
  say "=== seeds finished ==="
}

# ------------------------------------------------------------------- stage: bcos
stage_bcos() {
  guard_idle
  say "=== stage: bcos — is the B-cos head just HiResCAM? ==="
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import numpy as np, torch, os
from scipy import stats
from torch.utils.data import Subset
from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.evaluate import ante_hoc_attr
from pxai.explainers.posthoc import explain_posthoc

run = "roi477_bcos_120ep"
cfg = load_config(f"configs/generated/{run}.yaml"); cfg["device"] = "cuda"
dev = pick_device(cfg["device"]); loaders = build_loaders(cfg)
cfg["model"]["num_classes"] = len(loaders.classes)
m = build_model(cfg).to(dev)
m.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location=dev)["model"]); m.eval()

ds = loaders.test.dataset
base = ds.dataset if isinstance(ds, Subset) else ds
while isinstance(base, Subset):
    base = base.dataset
idxs = list(ds.indices) if isinstance(ds, Subset) else range(len(base.samples))
rng = np.random.default_rng(1337)
rows = []
for gi in rng.choice(len(idxs), 55, replace=False):
    x, y = base[idxs[gi]]
    x = x.unsqueeze(0).to(dev); t = torch.tensor([y], device=dev)
    with torch.enable_grad():
        a = ante_hoc_attr("bcos")(m, x, t).detach().cpu().numpy().ravel()
        b = explain_posthoc("hirescam", m, x, t)[0].detach().cpu().numpy().ravel()
    if a.size == b.size:
        rows.append((stats.pearsonr(a, b)[0], stats.spearmanr(a, b).correlation))
p = np.mean([r[0] for r in rows]); s = np.mean([r[1] for r in rows])
print(f"\n  n={len(rows)}   pearson {p:.4f}   spearman {s:.4f}")
print("  " + ("EQUIVALENT: the B-cos-style head's explanation IS a gradient-weighted"
              "\n  CAM. Report as a finding -- it questions whether this head provides a"
              "\n  distinct interpretability mechanism." if p > 0.95 else
              "DISTINCT: correlated but not identical; the deletion tie is a"
              "\n  coincidence of the metric, not of the maps."))
PY
  say "=== bcos finished ==="
}

# ------------------------------------------------------------------- stage: push
stage_push() {
  guard_idle
  say "=== stage: push — is best.pt post-push? ==="
  say "  measures drift between saved prototypes and their nearest EVAL-transform patch"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import numpy as np, torch, os
from torch.utils.data import Subset
from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model

for run in ["roi477_protopnet_120ep", "roi477_protopnet_s2337_120ep"]:
    if not os.path.exists(f"runs/{run}/best.pt"):
        continue
    cfg = load_config(f"configs/generated/{run}.yaml"); cfg["device"] = "cuda"
    dev = pick_device(cfg["device"]); loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    m = build_model(cfg).to(dev)
    m.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location=dev)["model"])
    m.eval()
    head = m.head; P = head.prototypes.shape[0]; ppc = P // len(loaders.classes)

    # EVAL transforms: the earlier 0/55 result compared against AUGMENTED views, so
    # nothing could match exactly. loaders.test uses the deterministic transform.
    best = torch.full((P,), float("inf"))
    with torch.no_grad():
        for x, y in loaders.test:
            z = head.add_on(m.features(x.to(dev)))
            B, D, H, W = z.shape
            zf = z.permute(0, 2, 3, 1).reshape(B, H * W, D)
            for c in torch.unique(y).tolist():
                sel = zf[(y == c).to(zf.device)].reshape(-1, D)
                if sel.numel() == 0:
                    continue
                for p in range(c * ppc, (c + 1) * ppc):
                    d = ((sel - head.prototypes[p].view(1, D)) ** 2).sum(1).min()
                    best[p] = min(best[p].item(), d.item())
    b = best.numpy(); nrm = float(head.prototypes.view(P, -1).norm(dim=1).mean())
    print(f"\n{run}")
    print(f"  min sq-distance to nearest same-class patch: "
          f"median {np.median(b):.5f}  max {b.max():.5f}")
    print(f"  as a fraction of prototype norm: {np.median(np.sqrt(b))/nrm:.3%}")
    print("  " + ("POST-PUSH: prototypes essentially ARE real patches; 'this looks "
                  "like that' holds." if np.median(np.sqrt(b))/nrm < 0.05 else
                  "DRIFTED: best.pt was saved between pushes, so prototypes are NEAR "
                  "but not AT\n  real patches. State this in Part I; it does not "
                  "affect the localisation result."))
PY
  say "=== push finished ==="
}

# ------------------------------------------------------------------ stage: faith
stage_faith() {
  guard_idle
  say "=== stage: faith — faithfulness under GRADIENT attribution ==="

  if ! python -c "import sys; sys.exit(0 if 'PXAI_GRADATTR' in open('pxai/evaluate.py').read() else 1)"; then
    say "  applying apply_gradattr_eval.py"
    python apply_gradattr_eval.py 2>&1 | tee -a "$MASTER"
  fi

  say "  verifying the flag bites (conc_pos should read ~5.5, not ~2.5)"
  PXAI_GRADATTR=1 python -u batch_visualise.py --runs roi477_protopnet_120ep --fast \
      --outdir /tmp/gchk --tsv /tmp/gchk.tsv --device cuda >/dev/null 2>&1 || true
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, statistics as st, os
p = "/tmp/gchk.tsv"
if os.path.exists(p):
    v = [float(r["conc_pos"]) for r in csv.DictReader(open(p), delimiter="\t")
         if r["method"] == "ours:protopnet" and r["conc_pos"] not in ("nan", "")]
    if v:
        m = st.mean(v)
        print(f"  conc_pos = {m:.2f}  ->  " +
              ("FLAG ACTIVE" if m > 4.0 else "** FLAG NOT BITING -- do not run the eval"))
PY

  for a in "${FAITH[@]}"; do
    [ -f "runs/$a/best.pt" ] || continue
    [ -f "runs/$a/results.json" ] && \
      cp "runs/$a/results.json" "runs/$a/results.json.native-attr"
    rm -f "runs/$a/.eval_complete"
  done

  printf '%s\n' "${FAITH[@]}" | xargs -P "$PAR_EVAL" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || exit 0
    echo "  start $a"
    PXAI_GRADATTR=1 PXAI_GRADATTR_SMOOTH=1 \
      python -u -m pxai.evaluate --config "configs/generated/$a.yaml" \
        --ckpt "runs/$a/best.pt" > "runs/$a/eval_gradattr.log" 2>&1 \
      && echo "  done  $a" || echo "  FAIL  $a"' 2>&1 | tee -a "$MASTER"

  say "  native vs gradient attribution, deletion (lower is better)"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import json, os
print(f"  {'run':<26}{'method':<20}{'native':>9}{'gradattr':>10}{'change':>9}")
for a in ("roi477_protopnet_120ep", "roi477_cbm_sup_120ep", "roi477_bcos_120ep"):
    n, g = f"runs/{a}/results.json.native-attr", f"runs/{a}/results.json"
    if not (os.path.exists(n) and os.path.exists(g)):
        continue
    dn, dg = json.load(open(n)), json.load(open(g))
    for meth in dn.get("methods", {}):
        if not meth.startswith("ours:"):
            continue
        x = dn["methods"][meth]["faithfulness"].get("deletion", float("nan"))
        y = dg.get("methods", {}).get(meth, {}).get("faithfulness", {}).get(
            "deletion", float("nan"))
        tag = "better" if y < x else "worse"
        print(f"  {a:<26}{meth:<20}{x:>9.4f}{y:>10.4f}{tag:>9}")
print("""
  BETTER -> the gradient read-out wins on localisation AND faithfulness; the
            recommendation in Part II is unambiguous.
  WORSE  -> localisation and faithfulness genuinely trade off, which is a more
            interesting finding than either alone. Say so plainly.""")
PY
  say "=== faith finished ==="
}

case "${1:-all}" in
  audit) stage_audit ;;
  seeds) stage_audit; stage_seeds ;;
  bcos)  stage_bcos ;;
  push)  stage_push ;;
  faith) stage_faith ;;
  all)   stage_audit; stage_seeds; stage_bcos; stage_push; stage_faith ;;
  *)     echo "usage: $0 [all|audit|seeds|bcos|push|faith]"; exit 2 ;;
esac

say "done. master log: $MASTER"
