# pxai evaluation harness — handoff notes

**Last updated:** 2026-07-26 (supersedes the 2026-07-24 version, kept as
`EVAL_HARNESS_NOTES_v1_2026-07-24.md.bak`)
**Scope:** `pxai/eval/faithfulness.py`, `pxai/evaluate.py`, `pxai/explainers/posthoc.py`,
`pxai/eval/cost.py`, plus `configs/default.yaml`
**Verified against:** Quantus 0.6.0, Captum 0.9.0, Python 3.12, `batch_size: 32`, `img_size: 224`
**Status:** eval harness CLOSED — two configs (`ref_blackbox_convnext`,
`A2_protopnet_mobilevit`) produce complete, NaN-free, five-axis faithfulness
results. Remaining work is training-time only (§7) and does not block the eval.

---

## 0. TL;DR of the whole thread

A black-box eval was running ~88 h/config (23 configs ≈ 83 GPU-days). Diagnosis found
the slowness was mostly wrong Quantus defaults, but also that several numbers it was
about to produce were **wrong** — three bugs that flattered the project's own method on
the exact axis the C3 interpretability paper depends on. Fixed all of them, cut runtime
~20×, and closed a subtle ProtoPNet sanity-check artifact that took a redesign to get
right. Model training itself is sound.

---

## 1. Runtime — where the 88 hours went

Per-batch, gradcam pass (8 batches × 32 images):

| metric | old time | cause |
|---|---|---|
| infidelity | 25.5 min | Quantus default `perturb_patch_sizes=[4]` → 31,360 fwd passes/batch |
| sanity_check (MPRT) | 64 s | ~81 re-explanations (one per randomised layer) |
| deletion / insertion / sensitivity | seconds | — |

LIME/KernelSHAP were catastrophic because MPRT × ~2.9 min/LIME-attribution ≈ 3h53m per
batch. Fixes: `infidelity_patch_size: 28` (~49× fewer passes), `skip_layers=True` on
MPRT (~81 attributions → 1), superpixel `feature_mask` on LIME/KernelSHAP, smaller image
budget for the two re-explaining metrics. Result: **~4 h/config**, later measured at
~1.5–2 h.

---

## 2. Correctness bugs (these changed the numbers)

All three pushed in the direction of the project's own method. By severity.

### 2.1 `ante_hoc_attr` explained a captured model
`ev = model.explain(x)` used the closure `model` instead of the argument `m`. MPRT hands
in randomised deepcopies; the closure discarded them → explanation invariant → Spearman
1.0 = maximal sanity-check FAILURE recorded for every ProtoPNet/CBM/B-cos run. Fix:
`m.explain(x)`. (18 of 23 configs affected.)

### 2.2 Two metric signs inverted in the aggregate
- **`sanity_check`** — MPRT's `correlation_spearman`: high = explanation ignored the
  weights = failed. Lower is better.
- **`insertion`** — Quantus `PixelFlipping` removes pixels most-relevant-first, i.e. a
  deletion-style curve despite the name. Lower is better.

Combined with 2.1, interpretable heads scored the *best* sanity value because they were
broken. Fixed via the explicit `METRIC_DIRECTION` table.

### 2.3 LIME/KernelSHAP had no `feature_mask`
Captum default = one feature per element: 150,528 features fit from 1,000 samples per
image. Unidentifiable Lasso → constant attribution columns → the `invalid value
encountered in divide` warnings. These are the headline baselines; their old scores were
noise. Fix: 16×16 superpixel `feature_mask` (256 features). Expect these baselines to get
STRONGER, which makes "we beat post-hoc" harder — the correct, honest outcome.

### 2.4 Other real fixes
- `normalised_aggregate` didn't normalise (averaged incompatible scales dominated by
  infidelity magnitude). Replaced by `normalised_aggregates()` — per-metric min-max
  across methods, sign-aligned. **Relative to the methods compared**, so the per-metric
  table must travel with the Pareto plot.
- `cost_report` computed FLOPs twice (duplicated `aten::gelu` in logs) and mapped `0.0`
  to `None`.
- `latency_cpu` stranded the model on CPU in-place — fixed with device save/restore.
- No incremental writes: `results.json` only written after all methods + trust layer, so
  a kill lost everything. Now `_write` after every method; `--resume` skips done methods.
  **Delete/rename stale `results.json` before the first run under new code — `--resume`
  can't tell old-format from new and will merge broken scores.**

---

## 3. Metric-definition changes (NOT bugs — disclose in the paper)

Both old and new values are defensible; they measure slightly different things.

| knob | value | note |
|---|---|---|
| `infidelity_patch_size` | 28 (was default 4) | coarser scale; NOT comparable to papers using the default — state the value |
| `sanity_skip_layers` | true | scores original-vs-final-randomisation-step, not the per-layer curve |
| `sanity_layer_order` | top_down | **accumulates** randomisation; the scored (final) step is the FULLY randomised model. (`independent` restores weights each step and randomises one layer — for ProtoPNet that only touched the classifier head and gave a false 1.0; hence top_down.) |
| sensitivity / sanity_check images | 64 (others 256) | same definition, noisier estimate, no bias — state the N |

For C3, a per-layer degradation curve (`sanity_skip_layers: false`) on 2–3 headline
models is a stronger figure than the single number. Expensive; headline models only.

---

## 4. The ProtoPNet sanity_check artifact (the hard one)

**Symptom:** after fixing 2.1/2.2, `ours:protopnet` still scored `sanity_check = 1.0` —
which reads to a reviewer as a sanity FAILURE for the interpretable method.

**Diagnosis (took two wrong turns; the probe settled it):**
- `diagnose_protopnet_maps.py` confirmed the trained-model maps have strong spatial
  contrast (mean std/|mean| ≈ 0.54, clear hotspots) → explanations ARE spatial, C3 claim
  holds, the sigmoid add-on is fine (matches Chen et al. exactly).
- `probe_sanity_1p0.py` reproduced MPRT's randomisation on the real MobileViT and showed
  the *randomised* map is near-constant (rel ≈ 0.0004) and, when prototypes are actually
  randomised, Spearman correctly collapses to ~0.09. So the 1.0 is a degeneracy artifact:
  MPRT correlates two near-constant arrays and returns a spurious ~1.0 instead of ~0.

**Fix:** per-sample near-constant guard in `faithfulness.py`. First attempt observed
MPRT's internal `explain_func` calls — worked on a toy model, FAILED on the real one
(order-dependent). Redesigned as `_randomised_flat_mask`: self-contained, independently
reproduces the fully-randomised model (exactly as the probe does), flags samples whose
randomised explanation is flat, scores those `0.0` (pass). Order- and model-independent.

**Verified:** corrects flat ProtoPNet to a pass; leaves a genuinely-structured explainer
untouched (cannot silently zero a method that is actually failing). On the real model:
`sanity_check = 0.0, collapse = 64` (all 64 samples corrected).

**Paper note:** standard MPRT does NOT randomise the prototype parameters — `ProtoHead.
prototypes` is a bare `nn.Parameter` with no `reset_parameters`, so only the backbone is
randomised. The honest sanity result comes from backbone randomisation. Optionally
strengthen by adding `ProtoHead.reset_parameters` (eval-only, no retrain) so the
prototypes are randomised too — a stricter test. Not required.

New `results.json` fields: `sanity_collapse_batches` (samples scored 0.0 via the guard),
`explainer_determinism`, `n_samples`, `failures`, `batches_used`, and top-level
`eval_setup`.

---

## 5. Config (`configs/default.yaml`, `eval:` block)

```yaml
eval:
  faithfulness: [deletion, insertion, infidelity, sensitivity, sanity_check]
  faithfulness_batches: 8
  faithfulness_metric_batches:
    sensitivity: 2
    sanity_check: 2
  faithfulness_params:
    infidelity_patch_size: 28
    sanity_skip_layers: true
    sanity_layer_order: top_down
  localisation: [pointing_game, energy_pointing_game, iou]
  cost: [params, flops, latency_cpu, model_size_mb]
  quantize_int8: false
```

Regenerate configs after editing: `python make_configs.py` (it deep-copies at generation
time, so generated configs won't see edits until regenerated).

---

## 6. Running it

Smoke test first (`faithfulness_batches: 1`, two explainers so the aggregate has spread),
into a throwaway `output_dir`:

```bash
python -u -m pxai.evaluate --config configs/generated/<name>.yaml \
  --ckpt runs/<name>/best.pt 2>&1 | tee smoke.log
```

Full run, backgrounded and hangup-proof:

```bash
nohup python -u -m pxai.evaluate --config configs/generated/<name>.yaml \
  --ckpt runs/<name>/best.pt > runs/<name>/eval.log 2>&1 &
```

`run_evals.sh` chains configs sequentially (no `&` between them = never parallel;
parallel corrupts results). Launch: `nohup ./run_evals.sh > runs/all_evals.log 2>&1 &`.

### Operational traps learned the hard way
- **`-u` is required.** stdout is block-buffered under nohup; without `-u`, `[faithfulness]
  ... failed` messages sit in a buffer for hours while tqdm streams to stderr.
- **`.pyc` caches bite.** Editing a file does nothing to a running process, and stale
  `__pycache__/*.pyc` can run instead of edited source. After any edit:
  `find pxai -path "*__pycache__*" -name "*.pyc" -delete`.
- **Two Python versions on this box** (3.12 and 3.14, separate caches/packages). Always
  launch with `python` (= /usr/bin/python, 3.12). Don't mix.
- **Shutting the laptop DOES drop SSH** → the server sends SIGHUP. `nohup` protects the
  job, not leaving the laptop on. `&` (not `nohup`) is what makes it Ctrl-C-proof — they
  cover different signals.
- **A paste where only the reader runs** (not the eval) silently reads a stale
  `results.json`. Delete the output dir first so the reader can't succeed on stale data.
- **`>` truncates** `eval.log` on re-run. Rename a log first if you want to keep it.

### Morning checks
```bash
cat runs/all_evals.log                 # two "exit code 0" + ALL DONE
python -c "
import json
for n in ['ref_blackbox_convnext','A2_protopnet_mobilevit']:
    r=json.load(open(f'runs/{n}/results.json')); print('===',n,'acc',round(r['accuracy'],4))
    for m,d in r['methods'].items():
        print(f'  {m:22} sanity={d[\"faithfulness\"][\"sanity_check\"]}  collapse={d.get(\"sanity_collapse_batches\")}')
"
```
Every method must show a real `sanity_check` (no NaN, no spurious 1.0).

---

## 7. Training-time TODOs (NOT blocking the eval; do before training the other 21 configs)

Real, confirmed from logs/code — but the model that trained is sound (both `best.pt` are
the true best epoch; prototypes ARE projected; test tracks val closely, no leakage
signal).

1. **ProtoPNet convergence.** Val acc still climbing at epoch 44 of 60; the black box
   plateaued at epoch 11. The reported ~6-point gap (0.910 vs 0.971) may be partly a
   shared-budget artifact. Retrain A2 at 120 epochs (or early-stopping) before citing that
   gap as the interpretability tax.

2. **Prototype push coverage.** `_push_prototypes(..., max_batches=30)` searches 960 of
   7,680 candidates (12.5%). Prototypes are still real patches, but not guaranteed best
   matches — a discrepancy from standard ProtoPNet (full-set search). Raise to cover the
   full loader on retrain, or state the budget in methods.

3. **Push class filtering.** In `_push_prototypes`, `y` is unpacked but never used, so a
   prototype can snap onto a patch from the WRONG species — bad for a clinical explanation.
   Add class filtering (`y`). Before trusting the prototype figure, measure what fraction
   of prototypes' nearest patches are same-class on the current `best.pt`.

4. **Fix once, inherit free.** Items 2–3 live in `_push_prototypes`; fix before training
   the other 21 configs and they all inherit it. Only A2 is trained so far.

---

## 8. What changed in the numbers (for anyone comparing to old results)

- `sanity_check` for all interpretable heads: was structurally 1.0, now real.
- `insertion`/`sanity_check` signs corrected in the aggregate.
- LIME/KernelSHAP attributions entirely (mask); expect them stronger.
- ProtoPNet `sanity_check`: 1.0 → 0.0 (artifact removed).
- **Unchanged:** accuracy, ECE, temperature, conformal coverage, set size, model weights.
  The "hold accuracy while adding three deployment axes" claim is unaffected.
- Old result files preserved as `*.old-harness`, `*.pre-sanityfix`, `*.stale-1p0`,
  `*.pre-nearconst`. Never mix old- and new-harness numbers in one table.

---

## 9. Companion files

- `diagnose_protopnet_maps.py` — trained-model sim_map contrast + heatmap figure. Genuine
  C3 evidence that explanations are spatial. Run at `--n 12` for a paper figure.
- `probe_sanity_1p0.py` — reproduces MPRT randomisation on the real model; the tool that
  settled the sanity artifact. Keep for future ProtoPNet-family debugging.
- `why_the_slow_run_was_not_better.pdf` — the "longer ≠ better" reasoning, if the
  cheap-settings choices are ever questioned.