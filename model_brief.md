# The 23 configs — what each one is, and what actually needs training

**Date:** 2026-07-27
**Source:** `make_configs.py` + `configs/default.yaml`, verified against `pxai/models/`,
`pxai/train.py`, `pxai/evaluate.py`, `pxai/explainers/`
**Bottom line:** 23 configs → **12 unique training runs**, of which 2 are done.
**8 configs are no-ops** (the flag they toggle is never read by any code).

---

## 0. First: the comparison error to avoid

ProtoPNet's `val 0.9933` is a **120-epoch** number. The black box's `0.9655` is a
**60-epoch** number. These are not comparable, and the difference between them is not
an interpretability result — it is a budget difference. Earlier in this project the
whole reason for the ProtoPNet retrain was that 60 epochs was too short; citing a
120-vs-60 pair as evidence reintroduces exactly that confound.

**Rule for every number in the ablation ladder: same epoch budget, or don't compare.**
Either retrain the black box at 120, or report both at 60. Do not mix.

---

## 1. The model heads (what is actually being compared)

| head | mechanism | `.explain()` returns | notes |
|---|---|---|---|
| **blackbox** | backbone → avgpool → linear | *(none)* | reference to beat; post-hoc explainers attach here |
| **protopnet** | learned prototypes, L2 distance → similarity maps → sparse linear | `sim_maps`, `proto_class` | Chen et al. add-on = Conv→ReLU→Conv→Sigmoid (matches paper) |
| **cbm** | avgpool → 16 sigmoid concepts → linear on concepts ONLY | `concepts`, `class_concept_contrib` | hard bottleneck; supports test-time concept intervention |
| **bcos** | 1×1 B-cos conv, `|cos|^(B-1)` scaling | `contrib_map` | head-only variant; backbone convs are NOT B-cos |

**Backbones** (A1 axis): `mobilevit_xs` (~2.3 M, the sub-10 MB target),
`efficientnet_lite0` (~4.6 M), `ghostnet_100` (~5.2 M), `convnext_tiny` (~28 M, heavy
reference — not lightweight).

---

## 2. All 23 configs

`T#` = which unique training run it maps to (see §3). Configs sharing a `T#` train an
**identical model** — they differ only in flags that are currently unread.

| # | config | backbone | head | what it varies | T# | status |
|---|---|---|---|---|---|---|
| 1 | `ref_blackbox_convnext` | convnext_tiny | blackbox | accuracy ceiling + post-hoc target | **T1** | trained 60ep |
| 2 | `A2_protopnet_mobilevit` | mobilevit_xs | protopnet | — (base config) | **T2** | trained 120ep ✅ |
| 3 | `A2_cbm_mobilevit` | mobilevit_xs | cbm | head = concept bottleneck | **T3** | — |
| 4 | `A2_bcos_mobilevit` | mobilevit_xs | bcos | head = B-cos | **T4** | — |
| 5 | `A1_mobilevit_xs` | mobilevit_xs | protopnet | backbone sweep | **T2** | ⚠ duplicate of #2 |
| 6 | `A1_efficientnet_lite0` | efficientnet_lite0 | protopnet | backbone sweep | **T5** | — |
| 7 | `A1_ghostnet_100` | ghostnet_100 | protopnet | backbone sweep | **T6** | — |
| 8 | `A1_convnext_tiny` | convnext_tiny | protopnet | backbone sweep (heavy) | **T7** | — |
| 9 | `A3_proto1` | mobilevit_xs | protopnet | 1 prototype/class | **T8** | — |
| 10 | `A3_proto3` | mobilevit_xs | protopnet | 3 prototypes/class | **T9** | — |
| 11 | `A3_proto5` | mobilevit_xs | protopnet | 5 prototypes/class | **T2** | ⚠ duplicate of #2 (default is 5) |
| 12 | `A3_proto10` | mobilevit_xs | protopnet | 10 prototypes/class | **T10** | — |
| 13 | `A3_concepts8` | mobilevit_xs | cbm | 8 concepts | **T11** | — |
| 14 | `A3_concepts16` | mobilevit_xs | cbm | 16 concepts | **T3** | ⚠ duplicate of #3 (default is 16) |
| 15 | `A3_concepts32` | mobilevit_xs | cbm | 32 concepts | **T12** | — |
| 16 | `A4_amortized_ep10` | convnext_tiny | blackbox | amortized explainer, 10 ep | **T1** | 🔴 **no-op** |
| 17 | `A4_amortized_ep30` | convnext_tiny | blackbox | amortized explainer, 30 ep | **T1** | 🔴 **no-op** |
| 18 | `A4_amortized_ep60` | convnext_tiny | blackbox | amortized explainer, 60 ep | **T1** | 🔴 **no-op** |
| 19 | `A5_distilled_amortized` | convnext_tiny | blackbox | distil SHAP from teacher | **T1** | 🔴 **no-op** |
| 20 | `A6_conformal_on` | mobilevit_xs | protopnet | conformal abstention ON | **T2** | 🔴 **no-op** |
| 21 | `A6_conformal_off` | mobilevit_xs | protopnet | conformal abstention OFF | **T2** | 🔴 **no-op** |
| 22 | `A7_int8_off` | mobilevit_xs | protopnet | fp32 | **T2** | 🔴 **no-op** |
| 23 | `A7_int8_on` | mobilevit_xs | protopnet | INT8 quantized | **T2** | 🔴 **no-op** |

---

## 3. The 12 unique training runs

| T# | backbone + head + hyperparams | serves configs | status |
|---|---|---|---|
| T1 | convnext_tiny + blackbox | 1, 16, 17, 18, 19 | trained **60ep** |
| T2 | mobilevit_xs + protopnet, ppc=5 | 2, 5, 11, 20, 21, 22, 23 | trained **120ep** ✅ |
| T3 | mobilevit_xs + cbm, nc=16 | 3, 14 | — |
| T4 | mobilevit_xs + bcos | 4 | — |
| T5 | efficientnet_lite0 + protopnet | 6 | — |
| T6 | ghostnet_100 + protopnet | 7 | — |
| T7 | convnext_tiny + protopnet | 8 | — |
| T8 | mobilevit_xs + protopnet, ppc=1 | 9 | — |
| T9 | mobilevit_xs + protopnet, ppc=3 | 10 | — |
| T10 | mobilevit_xs + protopnet, ppc=10 | 12 | — |
| T11 | mobilevit_xs + cbm, nc=8 | 13 | — |
| T12 | mobilevit_xs + cbm, nc=32 | 15 | — |

**10 unique models remain to train — not 21.** Training the duplicates would burn
~30 GPU-hours producing byte-identical checkpoints.

---

## 4. The no-ops — 8 configs that currently change nothing

Verified by grep across the whole package:

### A4 + A5 (configs 16–19): the amortized explainer is never called
`pxai/explainers/amortized.py` defines `AmortizedExplainer` and `fastshap_step`, but
**neither is imported by `train.py` or `evaluate.py`.** The only other mentions are
docstrings and a colour key in `plots/pareto.py`. So `explain.amortized.enabled`,
`train_epochs`, and `teacher_ckpt` are read by nothing. All four configs currently
reproduce `ref_blackbox_convnext` exactly.

**This matters:** the amortized explainer is contribution **C2** — the "Shapley
quality at 1 forward pass" claim, and the blue `amortized` points on the Pareto plot.
It is unimplemented in the pipeline. Wiring it is a real task, not a config change.

### A6 (configs 20–21): the conformal flag is never checked
`evaluate.py` runs the conformal block unconditionally — it never reads
`cfg["trust"]["conformal"]`. So `A6_conformal_on` and `A6_conformal_off` produce
identical `results.json`. To make A6 meaningful, gate the trust block on the flag.

### A7 (configs 22–23): quantization is never applied
`eval.quantize_int8` appears in no code path. `A7_int8_on` and `A7_int8_off` are
identical. To make A7 meaningful, apply `torch.ao.quantization` (or equivalent) before
the cost + faithfulness passes. Note the point of A7 is "does compression corrupt
explanations" — it needs quantization applied *and then* the same faithfulness suite
run, so it's an eval-path change.

---

## 5. Other gaps worth knowing

**CBM concept supervision is a no-op.** `train.py` calls
`concept_loss(c_logit, None)`, and `concept_loss` returns zero when the target is
`None`. So the 16 concepts are learned *without supervision* — they are arbitrary
latent dimensions, not the morphological concepts (operculum, shell texture, polar
knob) the proposal describes. The hard bottleneck is real (the label does depend only
on 16 sigmoid units), so it is still a bottleneck model — but you cannot currently
claim human-readable concepts. `cbm.concept_source: labelfree` is also unread
(`CBMHead` never receives it).

**B-cos is head-only.** `bcos.py` says so explicitly: for a fully weight-faithful
B-cos model the backbone convs must also be B-cos. The current head-only variant gives
a contribution map but not the paper's faithfulness guarantee.

**A1 uses the protopnet head** for the backbone sweep (per the comment in
`make_configs.py`: "swap after you pick the winner"). Fine, but state it — A1 measures
backbone × protopnet, not backbone alone.

---

## 6. Suggested plan

**Step 1 — fix the comparison baseline (do first).**
Retrain T1 (`ref_blackbox_convnext`) at 120 epochs so the headline
interpretable-vs-blackbox comparison is budget-matched. Until then, no
accuracy claim comparing them is valid. (~6 h at convnext speed.)

**Step 2 — decide the ladder's epoch budget from evidence, not assumption.**
Train T3 (cbm) and T4 (bcos) at 120 epochs next. If they also plateau late like
ProtoPNet did (0.917 → 0.993 between 60 and 120), set 120 for the whole ladder. If
they converge by 60, use 60 for the rest and save ~2 days. Two runs buys that answer.

**Step 3 — train the remaining unique models (T5–T12), 8 runs.**
At ~2 min/epoch on mobilevit and ~3 on convnext: roughly 1.5–2 days at 120 epochs,
under a day at 60.

**Step 4 — evaluate.** One eval per unique T#, not per config. ~1.5 h each ⇒ ~18 h.
Re-run T1's eval too: its current `results.json` predates the near-constant guard, so
it is not harness-consistent with A2's.

**Step 5 — decide what to do about A4/A5/A6/A7.**
Three options, and this is a scope call, not a technical one:
- **Implement them.** A6 and A7 are small (gate a flag; apply quantization). A4/A5
  (the amortized explainer, contribution C2) is a genuine implementation task.
- **Drop them from the ladder** and report 15 configs / 12 runs honestly.
- **Defer** A4/A5 to the 7–12 month stretch and implement only A6/A7 now.

Whichever you pick, **do not train the 8 no-op configs as they stand** — they produce
duplicate checkpoints and identical results tables.

---

## 7. Quick reference — what to run per unique model

```bash
# train (adjust epochs per the Step-2 decision)
nohup python -u -m pxai.train --config configs/generated/<name>.yaml \
  > runs/<name>/train.log 2>&1 &

# evaluate (after training)
nohup python -u -m pxai.evaluate --config configs/generated/<name>.yaml \
  --ckpt runs/<name>/best.pt > runs/<name>/eval.log 2>&1 &
```

One at a time — never parallel (corrupts results). See `EVAL_HARNESS_NOTES.md` §6 for
the operational traps (`-u`, pycache, stale `results.json`).