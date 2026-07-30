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


def build_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),   # stain/illumination robustness (ties to IPI-CVx aug theme)
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


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
    tf_tr = build_transforms(img, train=True)
    tf_ev = build_transforms(img, train=False)

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

    mk = lambda ds, sh: DataLoader(
        ds, batch_size=d["batch_size"], shuffle=sh,
        num_workers=d["num_workers"], pin_memory=True, drop_last=sh)
    return Loaders(mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), classes)
