"""Crop-aware annotation lookup, shared by the localisation probes.

Crop files are named `<stem>_<k>.jpg`; labels.json keys on `<stem>.jpg`, so a direct
basename lookup misses every crop -- which silently zeroed the night-3 probes
(0/0 prototypes, 0% annotation match).

AMBIGUITY: originals are `<class>_<number>.jpg` and crops `<class>_<number>_<k>.jpg`
-- both end in `_<digits>`. Resolved by consulting the annotation index.

On CROPPED data the egg fills 55-70% of the frame, so a pointing-game "hit" is close
to guaranteed by construction. Use the returned coverage as the chance baseline and
prefer the occlusion test.
"""
from __future__ import annotations
import json, os, re
from collections import defaultdict
import numpy as np

_CROP_RE = re.compile(r"^(?P<stem>.+)_(?P<k>\d+)$")


def load_coco(labels_path):
    with open(labels_path) as f:
        coco = json.load(f)
    meta = {im["id"]: (os.path.basename(im["file_name"]), im.get("width"), im.get("height"))
            for im in coco["images"]}
    out = defaultdict(lambda: {"w": None, "h": None, "boxes": []})
    for a in coco["annotations"]:
        if a["image_id"] not in meta:
            continue
        name, w, h = meta[a["image_id"]]
        out[name]["w"], out[name]["h"] = w, h
        out[name]["boxes"].append(tuple(a["bbox"]))
    return dict(out)


def parse_crop_name(path, ann=None):
    base = os.path.basename(path)
    if ann is not None and base in ann:
        return base, None
    stem, ext = os.path.splitext(base)
    m = _CROP_RE.match(stem)
    if not m:
        return base, None
    cand = m.group("stem") + ext
    if ann is not None and cand not in ann:
        return base, None
    return cand, int(m.group("k"))


def crop_rect(box, ow, oh, margin=0.20, square=True):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    w *= (1 + margin); h *= (1 + margin)
    if square:
        w = h = max(w, h)
    return (max(0, int(round(cx - w / 2.0))), max(0, int(round(cy - h / 2.0))),
            min(ow, int(round(cx + w / 2.0))), min(oh, int(round(cy + h / 2.0))))


def box_in_crop(path, ann, size, margin=0.20, square=True):
    orig, k = parse_crop_name(path, ann)
    rec = ann.get(orig)
    if rec is None or not rec["boxes"]:
        return None
    ow, oh = rec["w"], rec["h"]
    if not ow or not oh:
        return None
    m = np.zeros((size, size), dtype=bool)

    if k is None:
        sx, sy = size / float(ow), size / float(oh)
        for (x, y, w, h) in rec["boxes"]:
            x0 = int(np.clip(round(x * sx), 0, size - 1)); y0 = int(np.clip(round(y * sy), 0, size - 1))
            x1 = int(np.clip(round((x + w) * sx), 0, size)); y1 = int(np.clip(round((y + h) * sy), 0, size))
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = True
        return m

    if k >= len(rec["boxes"]):
        return None
    box = rec["boxes"][k]
    cx0, cy0, cx1, cy1 = crop_rect(box, ow, oh, margin, square)
    cw, chh = cx1 - cx0, cy1 - cy0
    if cw <= 0 or chh <= 0:
        return None
    x, y, w, h = box
    sx, sy = size / float(cw), size / float(chh)
    x0 = int(np.clip(round((x - cx0) * sx), 0, size - 1)); y0 = int(np.clip(round((y - cy0) * sy), 0, size - 1))
    x1 = int(np.clip(round((x - cx0 + w) * sx), 0, size)); y1 = int(np.clip(round((y - cy0 + h) * sy), 0, size))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m
