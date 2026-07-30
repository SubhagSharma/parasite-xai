"""
make_crop_configs.py — configs for the ROI-cropped dataset.

Three heads on the SAME cropped data, so the head comparison is finally made on
data without the background shortcut:

  crop_protopnet_120ep   interpretable, prototype-based
  crop_blackbox_120ep    linear head    -> is the residual confound model-agnostic?
  crop_bcos_120ep        interpretable, B-cos alignment

img_size stays 224: median crop is 262x262, so this is a mild downsample and the
numbers remain comparable with the whole-image models.

    python make_crop_configs.py --root ../Data/chula_crops
"""
import argparse, copy, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "configs", "generated")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Data/chula_crops")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    base_path = os.path.join(GEN, "A2_protopnet_mobilevit_120ep.yaml")
    with open(base_path) as f:
        base = yaml.safe_load(f)

    for kind, tag in (("protopnet", "crop_protopnet_120ep"),
                      ("blackbox",  "crop_blackbox_120ep"),
                      ("bcos",      "crop_bcos_120ep")):
        c = copy.deepcopy(base)
        c["data"]["root"] = args.root
        c["data"]["img_size"] = args.img_size
        c["backbone"]["name"] = "mobilevit_xs"
        c["model"]["kind"] = kind
        c["train"]["epochs"] = args.epochs
        c["output_dir"] = f"./runs/{tag}"
        p = os.path.join(GEN, f"{tag}.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print(f"wrote configs/generated/{tag}.yaml  ({kind} on {args.root})")

    print("\nall three share backbone=mobilevit_xs and the cropped root,")
    print("so differences are attributable to the HEAD alone.")

if __name__ == "__main__":
    main()
