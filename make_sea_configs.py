"""make_sea_configs.py — SEA configs derived from the roi477 base.

    python make_sea_configs.py                       # default base
    python make_sea_configs.py --base configs/generated/roi477_protopnet_120ep.yaml

Writes configs/generated/{PFX}_sea_*.yaml. Every arm changes ONE thing from
the arm above it, so the stride sweep is a controlled mechanism test rather
than a set of unrelated models.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "configs", "generated")
DEFAULT_BASE = os.path.join(OUT, "roi477_protopnet_120ep.yaml")


PFX = "roi477"
_SEEN: dict[str, str] = {}


def dump(cfg, name):
    """Write one config, and refuse to write two that differ only in name.

    DIRECTORY_MAP records that 11 of the original 23 ladder configs collapsed
    onto duplicates and 8 more were no-ops, which cost GPU nights on runs that
    could not differ. Everything except output_dir is hashed, so a duplicate is
    caught here instead of in the results table.
    """
    os.makedirs(OUT, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = f"./runs/{name}"
    body = copy.deepcopy(cfg)
    body.pop("output_dir", None)
    key = hashlib.md5(yaml.safe_dump(body, sort_keys=True).encode()).hexdigest()
    if key in _SEEN:
        raise SystemExit(
            f"DUPLICATE CONFIG: {name!r} is identical to {_SEEN[key]!r} "
            f"apart from output_dir.\nEither the two arms differ in a field "
            f"that is not in the config, or one of them is a no-op. "
            f"Fix make_sea_configs.py before running anything.")
    _SEEN[key] = name
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {os.path.relpath(path, HERE)}")


def sea(base, **kw):
    c = copy.deepcopy(base)
    c["model"]["kind"] = "sea"
    c["model"]["sea"] = {
        "stride": kw.get("stride", 8),
        "max_evidence_stride": kw.get("max_evidence_stride", 16),
        "dim": kw.get("dim", 64),
        "depth": kw.get("depth", 2),
        "readout": kw.get("readout", "mlp"),
        "context": kw.get("context", "none"),
        "context_adv": kw.get("context_adv", 0.0),
        "loss": {"conc": kw.get("conc", 0.0), "tv": kw.get("tv", 0.0),
                 "conc_on": kw.get("conc_on", "abs"),
                 "cross": kw.get("cross", 0.0), "probe": kw.get("probe", 0.0),
                 "tau": kw.get("tau", 0.10)},
    }
    if "backbone" in kw:
        c["backbone"]["name"] = kw["backbone"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--prefix", default="roi477")
    a = ap.parse_args()
    global PFX
    PFX = a.prefix
    with open(a.base) as f:
        base = yaml.safe_load(f)
    print(f"base: {a.base}  (img_size={base['data']['img_size']}, "
          f"epochs={base['train']['epochs']}, backbone={base['backbone']['name']})")

    # --- S1  the mechanism test: stride ONLY -------------------------------- #
    # Attribution priors are OFF here on purpose. This arm isolates spatial
    # resolution at fixed input size, fixed backbone, fixed everything else --
    # and because nothing in the loss ever sees a box, conc / peak / IoU stay
    # honest held-out metrics for it.
    for r in (32, 16, 8, 4):
        dump(sea(base, stride=r, conc=0.0, tv=0.0), f"{PFX}_sea_s{r}")

    # --- S2  the CAM anchor: proves the comparison is controlled ------------- #
    dump(sea(base, stride=32, readout="linear", context="none",
             conc=0.0, tv=0.0), f"{PFX}_sea_s32_cam")
    dump(sea(base, stride=4, readout="linear", context="none",
             conc=0.0, tv=0.0), f"{PFX}_sea_s4_cam")

    # --- S3  does the concentration prior add anything on real data? -------- #
    # Recommended setting from the synthetic sweep: conc=3, tau=0.10, on |phi|,
    # TV off. Run these against the matching S1 arm, which is the same model
    # with the prior removed.
    for r in (8, 4):
        dump(sea(base, stride=r, conc=3.0, tau=0.10),
             f"{PFX}_sea_s{r}_conc")

    # --- S4  factor ablations at the best stride (edit r once S1 lands) ------ #
    r = 8
    # context=none is the default, so the contrast arm is the one that turns
    # FiLM ON. Axiom check 4 measures FiLM's influence on phi at 0.195 relative
    # at init, which is not negligible -- this arm is what quantifies the cost.
    dump(sea(base, stride=r, conc=0.0, tv=0.0, context="film"),
         f"{PFX}_sea_s{r}_film")
    dump(sea(base, stride=r, conc=0.0, tv=0.0, readout="linear"),
         f"{PFX}_sea_s{r}_lin")
    dump(sea(base, stride=r, conc=0.0, tv=0.0, max_evidence_stride=8),
         f"{PFX}_sea_s{r}_smallrf")
    # TV is off by default because it HURT localisation on the synthetic task
    # (conc 7.15 -> 3.75). This arm is the ablation that documents that.
    dump(sea(base, stride=r, conc=0.0, tv=0.1), f"{PFX}_sea_s{r}_tv")
    # leakage probe: report the trained probe accuracy, target ~1/11 = 9.1%
    dump(sea(base, stride=r, conc=0.0, tv=0.0, context_adv=0.5, probe=0.5),
         f"{PFX}_sea_s{r}_advprobe")

    # --- S5  locality certificate needs a pure-conv backbone ---------------- #
    # mobilevit runs global attention, so no mobilevit arm can claim a bounded
    # receptive field. This arm can.
    dump(sea(base, stride=r, conc=0.0, tv=0.0, context="none",
             max_evidence_stride=8, backbone="efficientnet_lite0"),
         f"{PFX}_sea_s{r}_local_effnet")

    # --- S6  concentration strength sweep (only after S3 shows it helps) ---- #
    for lam in (0.3, 10.0):
        dump(sea(base, stride=r, conc=lam, tau=0.10),
             f"{PFX}_sea_s{r}_conc{lam}")


if __name__ == "__main__":
    main()
