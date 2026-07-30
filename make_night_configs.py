"""
make_night_configs.py — build the configs for tonight's serial run.

Produces / updates:
  1. configs/generated/ref_blackbox_convnext_120ep.yaml   (NEW)
       ConvNeXt-Tiny black box at 120 epochs. Exists solely to make the
       interpretable-vs-blackbox accuracy comparison BUDGET-MATCHED: ProtoPNet's
       0.9933 is a 120-epoch number and the current black box's 0.9655 is a
       60-epoch number, so they are not comparable.

  2. configs/generated/A2_protopnet_mobilevit_120ep.yaml  (UPDATED metrics only)
       Adds the three infidelity variants. Training is unchanged; best.pt is
       untouched.

Both configs get all THREE infidelity variants so one eval yields the full
comparison table:

  infidelity_raw     Quantus default (normalise=False) -- reproduces the current
                     results.json exactly, kept so old numbers stay comparable
  infidelity         Quantus + normalise=True
  infidelity_scaled  Quantus + Yeh et al. optimal scaling (the corrected metric)

    python make_night_configs.py
"""
import copy
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "configs", "generated")

METRICS = ["deletion", "insertion",
           "infidelity_raw", "infidelity", "infidelity_scaled",
           "sensitivity", "sanity_check"]

PARAMS = {
    "infidelity_patch_size": 28,
    "infidelity_n_perturb_samples": 10,
    "infidelity_normalise": True,      # applies to `infidelity` + `infidelity_scaled`
    "sanity_skip_layers": True,
    "sanity_layer_order": "top_down",
}


def apply_eval_block(c):
    ev = c.setdefault("eval", {})
    ev["faithfulness"] = list(METRICS)
    ev["faithfulness_batches"] = 8
    ev["faithfulness_metric_batches"] = {"sensitivity": 2, "sanity_check": 2}
    ev.setdefault("faithfulness_params", {}).update(PARAMS)
    return c


def main():
    # ---- 1. ConvNeXt black box, 120 epochs (budget parity with ProtoPNet) ----
    src = os.path.join(GEN, "ref_blackbox_convnext.yaml")
    with open(src) as f:
        c = yaml.safe_load(f)
    c = copy.deepcopy(c)
    c["train"]["epochs"] = 120
    c["output_dir"] = "./runs/ref_blackbox_convnext_120ep"
    apply_eval_block(c)
    dst = os.path.join(GEN, "ref_blackbox_convnext_120ep.yaml")
    with open(dst, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    print(f"wrote {os.path.relpath(dst, HERE)}")
    print(f"   backbone={c['backbone']['name']}  kind={c['model']['kind']}  "
          f"epochs={c['train']['epochs']}  -> {c['output_dir']}")

    # ---- 1b. B-cos head at 120 epochs (budget parity with ProtoPNet) ----
    bsrc = os.path.join(GEN, "A2_bcos_mobilevit.yaml")
    if os.path.exists(bsrc):
        with open(bsrc) as f:
            cb = yaml.safe_load(f)
        cb = copy.deepcopy(cb)
        cb["train"]["epochs"] = 120
        cb["output_dir"] = "./runs/A2_bcos_mobilevit_120ep"
        apply_eval_block(cb)
        bdst = os.path.join(GEN, "A2_bcos_mobilevit_120ep.yaml")
        with open(bdst, "w") as f:
            yaml.safe_dump(cb, f, sort_keys=False)
        print(f"wrote {os.path.relpath(bdst, HERE)}")
        print(f"   backbone={cb['backbone']['name']}  kind={cb['model']['kind']}  "
              f"epochs={cb['train']['epochs']}  -> {cb['output_dir']}")
    else:
        print(f"WARNING: {bsrc} not found -- run make_configs.py first")

    # ---- 2. A2 ProtoPNet: metrics only, training untouched ----
    a2 = os.path.join(GEN, "A2_protopnet_mobilevit_120ep.yaml")
    if os.path.exists(a2):
        with open(a2) as f:
            c2 = yaml.safe_load(f)
        apply_eval_block(c2)
        with open(a2, "w") as f:
            yaml.safe_dump(c2, f, sort_keys=False)
        print(f"updated {os.path.relpath(a2, HERE)}  (metrics only; best.pt untouched)")
    else:
        print(f"WARNING: {a2} not found -- run make_A2_retrain_config.py first")

    print(f"\ninfidelity variants in both configs: "
          f"{[m for m in METRICS if m.startswith('infid')]}")


if __name__ == "__main__":
    main()