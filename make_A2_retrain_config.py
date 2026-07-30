"""
make_A2_retrain_config.py — item 2's retrain config.

Generates a config that trains A2 (ProtoPNet + MobileViT) to convergence so you can
tell whether the ~6-point accuracy gap vs the black box (0.910 vs 0.971) is
architectural or just a starved training budget. Bundles the push fixes (items 3+4)
which are inherited automatically from the patched train.py.

Changes vs A2_protopnet_mobilevit.yaml:
  - epochs 60 -> 120                     (convergence: it was still climbing at ep44)
  - push_max_batches: null               (full-set prototype search, not 960 imgs)
  - output_dir -> runs/A2_protopnet_mobilevit_120ep
  - proto_push cadence kept at 10 -> pushes at 9,19,...,119 (12 pushes)

The class-filtered push needs NO config flag — it is now the only behaviour in
_push_prototypes. Run:

    python make_A2_retrain_config.py
    nohup python -u -m pxai.train \
      --config configs/generated/A2_protopnet_mobilevit_120ep.yaml \
      > runs/A2_protopnet_mobilevit_120ep_train.log 2>&1 &
"""
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "configs", "generated", "A2_protopnet_mobilevit.yaml")
DST = os.path.join(HERE, "configs", "generated", "A2_protopnet_mobilevit_120ep.yaml")


def main():
    with open(SRC) as f:
        c = yaml.safe_load(f)

    c["train"]["epochs"] = 120
    c["train"]["push_max_batches"] = None          # full training set per push
    c["output_dir"] = "./runs/A2_protopnet_mobilevit_120ep"

    # optional but sensible for a longer run: early-stopping knobs if train.py reads them
    c["train"].setdefault("early_stop_patience", 20)   # ignored if unsupported

    with open(DST, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    print(f"wrote {os.path.relpath(DST, HERE)}")
    print("  epochs:", c["train"]["epochs"],
          "| push coverage: full set | output:", c["output_dir"])
    print("\nlaunch:")
    print("  nohup python -u -m pxai.train \\")
    print(f"    --config configs/generated/A2_protopnet_mobilevit_120ep.yaml \\")
    print("    > runs/A2_protopnet_mobilevit_120ep_train.log 2>&1 &")


if __name__ == "__main__":
    main()