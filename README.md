# parasite_xai — Lightweight Interpretable DSS for Medical Microscopy

Reference implementation of the experimental protocol in
`../protocol_interpretable.html`: an inherently-interpretable, edge-deployable
clinical decision-support model for intestinal-parasite microscopy whose
explanations are **more faithful than post-hoc SHAP / LIME / Grad-CAM** on a
heavy black box, produced in a **single forward pass**, with **calibrated
conformal abstention**.

Lab context (reference only): Behera, Kumar, Ahlawat & Prasad, *IPI-CVx*,
IEEE IJCNN 2025 (DOI 10.1109/IJCNN64981.2025.11228975). This track runs
parallel to the lab's Meta-AI thread; IPI-CVx is cited as prior domain work,
not extended here.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Chula-ParasiteEgg-11 (11 species, ~11k images). Either pre-split
`root/{train,val,test}/<class>/*.jpg` or a flat `root/<class>/*.jpg` (auto
stratified split). Point `data.root` in `configs/default.yaml` at it.

## Run
```bash
# train an inherently-interpretable model (A2: protopnet | cbm | bcos)
python -m pxai.train    --config configs/default.yaml

# three-axis sweep -> results.json + faithfulness_vs_cost.png + risk_coverage.png
python -m pxai.evaluate --config configs/default.yaml --ckpt runs/exp001/best.pt
```

## What maps to what (protocol ↔ code)

| Protocol | Code |
|---|---|
| Inherently-interpretable core (PIP-Net/ProtoPNet, CBM, B-cos) | `pxai/models/{protopnet,cbm,bcos}.py` |
| Heavy black-box baseline (to beat) | `pxai/models/blackbox.py` |
| Post-hoc baselines (Grad-CAM, HiResCAM, IG, LIME, KernelSHAP) | `pxai/explainers/posthoc.py` |
| C2/H3 single-pass amortized SHAP explainer (FastSHAP-style) | `pxai/explainers/amortized.py` |
| §4a Faithfulness (deletion/insertion/infidelity/sanity) via Quantus | `pxai/eval/faithfulness.py` |
| §4c Cost (params/FLOPs/latency + passes-per-explanation) | `pxai/eval/cost.py` |
| §5 Trust: temperature scaling + conformal sets + risk-coverage | `pxai/eval/conformal.py` |
| The money figure (faithfulness × cost frontier) | `pxai/plots/pareto.py` |

## Ablation ladder (protocol §6)
- **A1** backbone footprint — set `backbone.name` (mobilevit_xs → convnext_tiny).
- **A2** interpretable head — set `model.kind`.
- **A3** prototype/concept sparsity — `protopnet.num_prototypes_per_class` / `cbm.num_concepts`.
- **A4** amortized explainer fidelity — `explain.amortized.train_epochs`, coalitions.
- **A5** explanation distillation from heavy teacher — `amortized.teacher_ckpt`.
- **A6** calibration/abstention on/off — `trust.conformal`.
- **A7** INT8 quantization vs faithfulness — `eval.quantize_int8`.

## Status / TODO
- [x] Backbones, three interpretable heads, black box, post-hoc baselines, amortized explainer, faithfulness/cost/trust eval, Pareto plot. All modules byte-compile.
- [ ] Wire concept labels for CBM supervision (`_compute_loss` in `train.py`).
- [ ] Localisation eval (pointing-game / IoU) — needs `data.boxes_dir`.
- [ ] Amortized-explainer train loop (`fastshap_step` is implemented; add the driver).
- [ ] ONNX INT8 export path for the A7 ablation + on-device latency.

## Caveats
- `evaluate.py` runs post-hoc baselines on the interpretable model for a smoke
  test; for the real comparison train `model.kind: blackbox` (convnext_tiny) and
  point the post-hoc explainers at *that* checkpoint.
- HiResCAM's faithfulness guarantee is architecture-specific — verify it holds
  for your chosen backbone's last conv.
- Faithfulness metrics disagree (the "disagreement problem"); always report
  per-metric **and** the normalised aggregate, never a single cherry-picked number.
