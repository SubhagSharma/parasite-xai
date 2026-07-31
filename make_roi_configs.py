"""
make_roi_configs.py — configs for the scale-normalised ROI dataset.

TEN configs, all cloned from A2_protopnet_mobilevit_120ep.yaml, so every field
except data.root, backbone.name, model.kind, output_dir (and num_workers on the
one whole-image run) is byte-identical to the existing 120-epoch family:
img_size 224, batch 32, 120 epochs, lr 3e-4, wd 1e-4, warmup 3, seed 1337, amp on.

  HEAD SWEEP, w477, mobilevit_xs        roi477_{blackbox,protopnet,bcos,cbm}_120ep
  WINDOW CONTROL                        roi679_blackbox_120ep
  BACKBONE SWEEP (RQ4), w477, blackbox  roi477_{effnetlite0,ghostnet,convnext}_120ep
                                        + roi477_blackbox serves as mobilevit_xs
  CBM ON THE OTHER TWO DATA VERSIONS    crop_cbm_120ep, whole_cbm_120ep

WHY ONE WINDOW FOR THE HEAD SWEEP.
The dry run gave no flat choice: retention at 477 px is 82.5% (Taenia) to 100%
(H. nana), at 679 px 67.5% to 87.5%. 477 wins on overall retention (91.2% vs
75.9%) and on class spread, and still clears the 99th-percentile egg at 407 px.
Training four heads on both windows discards half the work once a window is
chosen, so 679 gets one blackbox run as a control instead. If it wins, three
retrains -- paid only if the control says so.

WHY num_workers=16 ON whole_cbm ONLY.
Whole images are I/O bound decoding 1280x960 JPEGs; at nw=4 that one run costs 4h
on a 256-CPU box. Worker count changes neither batch composition nor ordering,
but each worker seeds its own augmentation RNG, so this run is NOT exactly
seed-comparable with the existing whole-image runs -- treat it as a reseed.
Pass --whole-workers 4 for strict comparability and add ~2.8h.

CBM CAVEAT.
pxai/train.py:81 calls concept_loss(c_logit, None), which returns zero. The 16
concepts are unsupervised latent dimensions, not named morphological concepts.
These runs answer "what does a 16-d hard bottleneck cost in accuracy" and give NO
interpretability result. If ../Data/Chula-ParasiteEgg-11/concepts.csv holds real
annotations, wiring it in beats anything here.

    python make_roi_configs.py
"""
import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "configs", "generated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-a", default="../Data/chula_roi2_w477")
    ap.add_argument("--roi-b", default="../Data/chula_roi2_w679")
    ap.add_argument("--crops", default="../Data/chula_crops")
    ap.add_argument("--whole", default="../Data/Chula-ParasiteEgg-11/data")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--whole-workers", type=int, default=16)
    a = ap.parse_args()

    with open(os.path.join(GEN, "A2_protopnet_mobilevit_120ep.yaml")) as f:
        base = yaml.safe_load(f)

    E = a.epochs
    jobs = [   # (name, backbone, head, root, num_workers override)
        (f"roi477_blackbox_{E}ep",    "mobilevit_xs",       "blackbox",  a.roi_a, None),
        (f"roi477_protopnet_{E}ep",   "mobilevit_xs",       "protopnet", a.roi_a, None),
        (f"roi477_bcos_{E}ep",        "mobilevit_xs",       "bcos",      a.roi_a, None),
        (f"roi477_cbm_{E}ep",         "mobilevit_xs",       "cbm",       a.roi_a, None),
        (f"roi679_blackbox_{E}ep",    "mobilevit_xs",       "blackbox",  a.roi_b, None),
        (f"roi477_effnetlite0_{E}ep", "efficientnet_lite0", "blackbox",  a.roi_a, None),
        (f"roi477_ghostnet_{E}ep",    "ghostnet_100",       "blackbox",  a.roi_a, None),
        (f"roi477_convnext_{E}ep",    "convnext_tiny",      "blackbox",  a.roi_a, None),
        (f"crop_cbm_{E}ep",           "mobilevit_xs",       "cbm",       a.crops, None),
        (f"whole_cbm_{E}ep",          "mobilevit_xs",       "cbm",       a.whole,
         a.whole_workers),
    ]

    for name, bb, kind, root, nw in jobs:
        c = copy.deepcopy(base)
        c["data"]["root"] = root
        c["data"]["img_size"] = a.img_size
        if nw is not None:
            c["data"]["num_workers"] = nw
        c["backbone"]["name"] = bb
        c["model"]["kind"] = kind
        c["train"]["epochs"] = E
        c["output_dir"] = f"./runs/{name}"
        with open(os.path.join(GEN, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        extra = f"  nw={nw}" if nw is not None else ""
        print(f"  {name:<28} {bb:<20} {kind:<10} {root}{extra}")

    print(f"\n{len(jobs)} configs written.")
    print("roi477_blackbox is BOTH the head-sweep baseline and the mobilevit_xs")
    print("entry in the backbone sweep, so RQ4 gets four backbones on identical data.")


if __name__ == "__main__":
    main()
