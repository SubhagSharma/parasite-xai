"""
make_aug_configs.py — augmentation sweep to kill the residual context shortcut.

After ROI cropping, the model still scores 0.2494 with the egg masked out, against a
chance level of 0.0909. So ~2.7x chance of the prediction still comes from something
other than the parasite. Cropping removed the border and background debris; what
survives a crop is GLOBAL APPEARANCE — colour cast, illumination, focus.

The existing augmentation is ColorJitter(0.2, 0.2, 0.2), which leaves HUE at 0. Hue
is exactly the axis that differs between microscopes and stains, so the one cue most
likely to be carrying the residual shortcut is the one never being jittered.

Four levels, increasingly aggressive about destroying colour:

  aug0_current   what we have now                      -> egg-masked 0.2494 (measured)
  aug1_hue       + hue jitter 0.15, vflip, rot 180     -> attacks colour cast directly
  aug2_strong    + jitter 0.4, hue 0.25, p(gray)=0.3,
                   p(blur)=0.3                         -> attacks colour AND focus
  aug3_gray      ALL images grayscale, train AND test  -> removes colour entirely

aug3 is the decisive one. Parasite eggs are identified by morphology (size, shape,
shell texture, operculum, polar knobs), so colour is largely stain artifact. If
accuracy holds under grayscale while the egg-masked score falls toward chance, colour
was the shortcut and this is the fix.

    python make_aug_configs.py --root ../Data/chula_crops --head protopnet
"""
import argparse, copy, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "configs", "generated")

LEVELS = {
    "aug0_current": {
        "hflip": True, "vflip": False, "rotation": 15,
        "color_jitter": [0.2, 0.2, 0.2, 0.0],
        "random_grayscale": 0.0, "blur_p": 0.0, "to_grayscale": False,
    },
    "aug1_hue": {
        "hflip": True, "vflip": True, "rotation": 180,
        "color_jitter": [0.3, 0.3, 0.3, 0.15],
        "random_grayscale": 0.0, "blur_p": 0.0, "to_grayscale": False,
    },
    "aug2_strong": {
        "hflip": True, "vflip": True, "rotation": 180,
        "color_jitter": [0.4, 0.4, 0.4, 0.25],
        "random_grayscale": 0.3, "blur_p": 0.3, "blur_sigma": [0.1, 2.0],
        "to_grayscale": False,
    },
    "aug3_gray": {
        "hflip": True, "vflip": True, "rotation": 180,
        "color_jitter": [0.3, 0.3, 0.0, 0.0],   # saturation/hue meaningless once gray
        "random_grayscale": 0.0, "blur_p": 0.3, "blur_sigma": [0.1, 2.0],
        "to_grayscale": True,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Data/chula_crops")
    ap.add_argument("--head", default="protopnet")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--prefix", default="crop")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers PER JOB. With --parallel 4 keep this "
                         "at ~ncpu/4 so the jobs do not fight over CPUs.")
    args = ap.parse_args()

    src = os.path.join(GEN, "crop_protopnet_120ep.yaml")
    if not os.path.exists(src):
        raise SystemExit(f"need {src} first")
    with open(src) as f:
        base = yaml.safe_load(f)

    for name, aug in LEVELS.items():
        c = copy.deepcopy(base)
        c["data"]["root"] = args.root
        c["data"]["augment"] = aug
        # batch_size is deliberately NOT touched: aug0_current must reproduce the
        # existing crop model's 0.2494, which trained at batch 32.
        c["data"]["num_workers"] = args.workers
        c["data"]["persistent_workers"] = True
        c["data"]["prefetch_factor"] = 6
        c["model"]["kind"] = args.head
        c["train"]["epochs"] = args.epochs
        tag = f"{args.prefix}_{args.head}_{name}"
        c["output_dir"] = f"./runs/{tag}"
        with open(os.path.join(GEN, f"{tag}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        j = aug["color_jitter"]
        print(f"wrote {tag:34} hue={j[3]:<5} gray_p={aug['random_grayscale']:<4} "
              f"blur_p={aug['blur_p']:<4} all_gray={aug['to_grayscale']}")

    print("\nthe number to watch is EGG-MASKED accuracy (chance = 0.0909).")
    print("current crop model: 0.2494. lower is better.")


if __name__ == "__main__":
    main()
