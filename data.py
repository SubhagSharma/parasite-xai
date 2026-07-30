"""Data loading for parasite-microscopy DSS experiments.

Primary testbed: Chula-ParasiteEgg-11 (11 species, ~11k images).
Expected on-disk layout (ImageFolder-style)::

    root/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      test/<class_name>/*.jpg

If only a flat `root/<class_name>/*.jpg` exists, `build_loaders` will make a
stratified train/val/test split using the ratios in the config.

Optional localisation ground truth (bounding boxes / masks) lives under
`boxes_dir` and is loaded lazily only by the localisation evaluator, so the
classification path never depends on it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ImageNet stats — backbones are ImageNet-pretrained (timm default).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# Default augmentation — reproduces the original hard-coded behaviour exactly, so
# configs without an `augment` block are unaffected.
#
# NOTE the original ColorJitter(0.2, 0.2, 0.2) leaves HUE at its default of 0. Hue is
# precisely the axis that differs between microscopes and stains (bluish vs yellowish
# cast), and colour cast is the residual shortcut left after ROI cropping: with the
# egg masked out, the cropped model still scores 0.2494 against a 0.0909 chance level.
# Jittering brightness/contrast/saturation does not touch it.
_DEFAULT_AUG = {
    "hflip": True,
    "vflip": False,
    "rotation": 15,
    "color_jitter": [0.2, 0.2, 0.2, 0.0],   # brightness, contrast, saturation, HUE
    "random_grayscale": 0.0,                # p of dropping colour for one sample
    "blur_p": 0.0,                          # p of Gaussian blur (focus robustness)
    "blur_sigma": [0.1, 1.5],
    "to_grayscale": False,                  # force ALL images grayscale (kills colour
                                            # as a usable cue entirely)
}


def build_transforms(img_size: int, train: bool, augment: dict | None = None):
    """Eval transform is fixed; train transform is driven by the `augment` dict.

    to_grayscale applies to BOTH splits — if colour is removed at training time it
    must be removed at test time too, or the model sees a distribution it never
    trained on.
    """
    a = {**_DEFAULT_AUG, **(augment or {})}
    gray = [transforms.Grayscale(num_output_channels=3)] if a["to_grayscale"] else []

    if not train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *gray,
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    ops = [transforms.Resize((img_size, img_size))]
    if a["hflip"]:
        ops.append(transforms.RandomHorizontalFlip())
    if a["vflip"]:
        # eggs have no canonical orientation in a microscope field, so this is free
        ops.append(transforms.RandomVerticalFlip())
    if a["rotation"]:
        ops.append(transforms.RandomRotation(a["rotation"]))
    cj = a["color_jitter"]
    if cj and any(v > 0 for v in cj):
        b, c, sat, hue = (list(cj) + [0.0] * 4)[:4]
        ops.append(transforms.ColorJitter(b, c, sat, hue))
    if a["random_grayscale"] > 0:
        ops.append(transforms.RandomGrayscale(p=a["random_grayscale"]))
    if a["blur_p"] > 0:
        k = max(3, (img_size // 32) * 2 + 1)
        ops.append(transforms.RandomApply(
            [transforms.GaussianBlur(k, sigma=tuple(a["blur_sigma"]))], p=a["blur_p"]))
    ops += gray
    ops += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    classes: list[str]


def _has_split_dirs(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, s)) for s in ("train", "val", "test"))


def build_loaders(cfg) -> Loaders:
    d = cfg["data"]
    root, img = d["root"], d["img_size"]
    aug = d.get("augment")            # None -> original hard-coded behaviour
    tf_tr = build_transforms(img, train=True, augment=aug)
    tf_ev = build_transforms(img, train=False, augment=aug)

    if _has_split_dirs(root):
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), tf_tr)
        val_ds = datasets.ImageFolder(os.path.join(root, "val"), tf_ev)
        test_ds = datasets.ImageFolder(os.path.join(root, "test"), tf_ev)
        classes = train_ds.classes
    else:
        # Flat folder -> stratified split. Build twice so train gets aug transforms.
        full_tr = datasets.ImageFolder(root, tf_tr)
        full_ev = datasets.ImageFolder(root, tf_ev)
        classes = full_tr.classes
        targets = [y for _, y in full_tr.samples]
        idx = list(range(len(targets)))
        tr_idx, tmp_idx = train_test_split(
            idx, test_size=d["val_split"] + d["test_split"],
            stratify=targets, random_state=cfg["seed"])
        rel = d["test_split"] / (d["val_split"] + d["test_split"])
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=rel,
            stratify=[targets[i] for i in tmp_idx], random_state=cfg["seed"])
        train_ds = Subset(full_tr, tr_idx)
        val_ds = Subset(full_ev, val_idx)
        test_ds = Subset(full_ev, test_idx)

    # persistent_workers keeps the worker processes alive between epochs. With 120
    # epochs that removes 120 rounds of process spawn + dataset re-open, which is a
    # large fraction of the per-epoch cost on a small dataset. prefetch_factor lets
    # each worker stage batches ahead so the GPU is not left waiting on JPEG decode.
    # Neither changes what the model sees -- ordering and augmentation are unaffected.
    nw = int(d.get("num_workers", 4))
    kw = {}
    if nw > 0:
        kw["persistent_workers"] = bool(d.get("persistent_workers", True))
        kw["prefetch_factor"] = int(d.get("prefetch_factor", 4))
    mk = lambda ds, sh: DataLoader(
        ds, batch_size=d["batch_size"], shuffle=sh,
        num_workers=nw, pin_memory=True, drop_last=sh, **kw)
    return Loaders(mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), classes)
