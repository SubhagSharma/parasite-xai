"""
make_ablation_configs.py — fill the missing cells of the head x backbone 2x2.

The current accuracy claim compares MobileViT-XS + interpretable head (0.9921)
against ConvNeXt-Tiny + black box (0.9709). Those differ in TWO variables, so the
+2.12 point gap cannot be attributed to the head. The 2x2:

                     black box head      interpretable head
    ConvNeXt-Tiny        0.9709  (have)      A1_convnext_tiny_120ep   (missing)
    MobileViT-XS     blackbox_mobilevit  (missing)      0.9921  (have)

Filling EITHER missing cell isolates the head effect. Filling both gives the full
factorial (main effect of head, main effect of backbone, interaction).

Writes:
  configs/generated/blackbox_mobilevit_120ep.yaml    (cheaper, more direct control)
  configs/generated/A1_convnext_tiny_120ep.yaml      (completes the 2x2)

Both at 120 epochs to stay budget-matched with everything else.

    python make_ablation_configs.py
"""
import copy
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "configs", "generated")


def main():
    base_path = os.path.join(GEN, "A2_protopnet_mobilevit_120ep.yaml")
    if not os.path.exists(base_path):
        raise SystemExit(f"need {base_path} first (run make_night_configs.py)")
    with open(base_path) as f:
        base = yaml.safe_load(f)

    # ---- cell 1: MobileViT-XS + BLACK BOX -------------------------------
    # Same backbone as our interpretable models, linear head. If this also scores
    # ~0.99, the gain came from the BACKBONE and the interpretability claim is
    # unsupported. If it scores ~0.97, the interpretable head is doing real work.
    c = copy.deepcopy(base)
    c["backbone"]["name"] = "mobilevit_xs"
    c["model"]["kind"] = "blackbox"
    c["train"]["epochs"] = 120
    c["output_dir"] = "./runs/blackbox_mobilevit_120ep"
    p1 = os.path.join(GEN, "blackbox_mobilevit_120ep.yaml")
    with open(p1, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    print(f"wrote {os.path.relpath(p1, HERE)}")
    print(f"   mobilevit_xs + blackbox, 120 ep -> {c['output_dir']}")

    # ---- cell 2: ConvNeXt-Tiny + PROTOPNET ------------------------------
    # Same backbone as the black box reference, interpretable head. Directly
    # answers "does the interpretable head cost accuracy on this backbone?"
    c2 = copy.deepcopy(base)
    c2["backbone"]["name"] = "convnext_tiny"
    c2["model"]["kind"] = "protopnet"
    c2["train"]["epochs"] = 120
    c2["output_dir"] = "./runs/A1_convnext_tiny_120ep"
    p2 = os.path.join(GEN, "A1_convnext_tiny_120ep.yaml")
    with open(p2, "w") as f:
        yaml.safe_dump(c2, f, sort_keys=False)
    print(f"wrote {os.path.relpath(p2, HERE)}")
    print(f"   convnext_tiny + protopnet, 120 ep -> {c2['output_dir']}")

    print("\nboth inherit the 7-metric eval block, but for the 2x2 you only need")
    print("ACCURACY -- use eval_accuracy_only.py (minutes) rather than the full")
    print("faithfulness sweep (hours).")


if __name__ == "__main__":
    main()
