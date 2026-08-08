"""
make_letterbox_config.py — the anisotropy control, without touching anything existing.

THE DEFECT (technical report §3.3)
----------------------------------
`pxai/data.py` uses `transforms.Resize((S, S))`. A two-tuple forces a square by
STRETCHING, so a circular egg emerges elliptical with eccentricity set entirely by the
source aspect ratio:

    1920x756  -> 2.54x horizontal        3024x4032 -> 1.33x vertical
    1280x672  -> 1.90x horizontal        2117x4032 -> 1.90x vertical
    1920x1080 -> 1.78x horizontal        1714x3264 -> 1.90x vertical

A 4.84x spread across the 13 native sizes -- larger than the 1.74x acquisition SCALE
spread that was corrected. Egg shape is diagnostic (barrel-shaped Trichuris vs round
Taenia), and it is being distorted by a transform determined by which camera was used.

WHICH ARM IS AFFECTED
    whole images   YES, uncorrected -- resized straight from the native aspect
    crops          no  -- make_crops.py squares the region first
    roi477/679     no  -- make_unified_roi_v2.py crops a square window

So it bites only the whole-image arm, which is the BASELINE the ROI is compared against.
Anisotropy is an acquisition fingerprint, so it inflates the apparent shortcut in the
baseline, which makes the reported "ROI removes 45-69% of the shortcut" an UPPER BOUND
(report §5.5.2). This control turns that bound into a number.

THE FIX
Letterbox: pad to square with the dataset mean colour, then resize. Preserves both
shape and field of view. Cost: constant-coloured borders, which are themselves a weak
source cue and must be reported as such -- a letterboxed 1920x756 image has far more
border than a letterboxed 896x960 one.

NOTHING IS OVERWRITTEN
  * a NEW transform class in pxai/letterbox.py -- data.py is untouched
  * a NEW config `whole_blackbox_lb_120ep` writing to a NEW run directory
  * the existing `blackbox_mobilevit_120ep` checkpoint, results.json and every TSV row
    stay exactly as they are; this arm is an ADDITIONAL data point, not a replacement

READING THE RESULT
Compare egg-masked accuracy against `blackbox_mobilevit_120ep` = 0.6806:

    letterboxed well BELOW 0.68  -> part of the whole-image shortcut was anisotropy.
                                    The ROI's 45-69% shrinks toward its true value and
                                    §5.5 needs revising downward.
    letterboxed NEAR 0.68        -> anisotropy was not contributing. 45-69% stands as
                                    measured and the §5.5.2 caveat can be discharged.

    python make_letterbox_config.py
"""
import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "configs", "generated")

LETTERBOX_SRC = '''"""Aspect-preserving square pad. NEW FILE -- pxai/data.py is not modified."""
from PIL import Image
import torchvision.transforms.functional as TF


class LetterboxSquare:
    """Pad the shorter side to make the image square, THEN it can be resized safely.

    `transforms.Resize((S, S))` stretches, which distorts object shape by an amount set
    by the source aspect ratio -- a 4.84x spread across this dataset's 13 native sizes.
    Padding first makes the subsequent resize isotropic, so a circle stays a circle.

    fill defaults to the ImageNet mean in 8-bit, matching the value the normalisation
    maps to zero, so the border is neutral after normalisation rather than a black bar.
    """

    def __init__(self, fill=(124, 116, 104)):
        self.fill = tuple(int(f) for f in fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        left, top = (s - w) // 2, (s - h) // 2
        return TF.pad(img, [left, top, s - w - left, s - h - top], fill=self.fill)

    def __repr__(self):
        return f"{type(self).__name__}(fill={self.fill})"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="blackbox_mobilevit_120ep",
                    help="the whole-image run this controls for")
    ap.add_argument("--name", default="whole_blackbox_lb_120ep")
    ap.add_argument("--seeds", default="1337")
    a = ap.parse_args()

    lb = os.path.join(HERE, "pxai", "letterbox.py")
    if os.path.exists(lb):
        print(f"  pxai/letterbox.py exists, left alone")
    else:
        with open(lb, "w") as f:
            f.write(LETTERBOX_SRC)
        print(f"  wrote pxai/letterbox.py")

    src = os.path.join(GEN, f"{a.src}.yaml")
    if not os.path.exists(src):
        raise SystemExit(f"source config not found: {src}")
    base = yaml.safe_load(open(src))

    for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
        c = copy.deepcopy(base)
        c["seed"] = seed
        c["data"]["letterbox"] = True          # read by the data.py patch
        name = a.name if seed == 1337 else f"{a.name.replace('_120ep','')}_s{seed}_120ep"
        c["output_dir"] = f"./runs/{name}"
        with open(os.path.join(GEN, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print(f"  {name:<34} seed={seed}  letterbox=True  root={c['data']['root']}")

    print(f"""
Everything else matches {a.src} exactly, so any difference is the transform alone.

NOT OVERWRITTEN: runs/{a.src}/ keeps its checkpoint, results.json and every TSV row.
This is an additional arm.

NEXT
  python apply_letterbox.py          # adds an opt-in branch to data.py, default off
  python -u preflight_learns.py --config configs/generated/{a.name}.yaml --device cuda
""")


if __name__ == "__main__":
    main()
