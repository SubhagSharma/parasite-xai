"""make_family_configs.py — three arms for the family-shared attention ablation.

  roi477_fam        fix 1 only            family-shared attention, no polar prior
  roi477_fam_polar  fix 1 + fix 2         + the DPDx polar prior on the 4 singletons
  dino_fam_polar    fix 1 + fix 2 + SSL   on the backbone that already reached 12/12
                                          cross-species consistency

Every arm derives from roi477_parts_120ep, so concept table, supervision and bottleneck
are identical to the measured baseline. The only differences are the head and the prior.

WHY THREE
  fam vs parts        isolates family-shared attention (architectural, dataset-agnostic)
  fam_polar vs fam    isolates the domain prior
  dino_fam_polar      combines everything with the backbone that crossed the resolution
                      threshold -- the arm most likely to work, and the least clean to
                      attribute if it does

Report the ablation, not just the best arm.
"""
import copy
import os

import yaml

GEN = "configs/generated"
base = yaml.safe_load(open(f"{GEN}/roi477_parts_120ep.yaml"))
cp = base["model"].get("concept_parts", {})
csv_path = cp.get("concepts_csv", "../Data/Chula-ParasiteEgg-11/concepts_v3.csv")

ARMS = [
    ("roi477_fam_120ep",       dict(w_polar=0.0), None),
    ("roi477_fam_polar_120ep", dict(w_polar=0.5), None),
    ("dino_fam_polar_120ep",   dict(w_polar=0.5), "dinov2_vits14"),
]

for name, opts, backbone in ARMS:
    c = copy.deepcopy(base)
    c["model"]["kind"] = "family_parts"
    c["model"]["family_parts"] = {
        "concepts_csv": csv_path,
        "slot_dim": cp.get("slot_dim", 128),
        "w_compact": cp.get("w_compact", 0.1),
        "bottleneck": cp.get("bottleneck", True),
        **opts,
    }
    if backbone:
        if not os.path.exists("pxai/models/dino_backbone.py"):
            print(f"  SKIP {name}: pxai/models/dino_backbone.py missing")
            continue
        c["backbone"]["name"] = backbone
        c["backbone"]["pretrained"] = True
    c["output_dir"] = f"./runs/{name}"
    yaml.safe_dump(c, open(f"{GEN}/{name}.yaml", "w"), sort_keys=False)
    print(f"  {name:<26} w_polar={opts['w_polar']:<5} "
          f"backbone={backbone or c['backbone']['name']}")

print("""
  baseline for comparison: roi477_parts_120ep  ->  2/12 anatomically consistent
                           dino_parts_120ep    -> 12/12 consistent but NOT anatomical

  the question these arms answer: does tying a family to ONE attention map, with a
  softmax over its mutually exclusive values, push that map onto the structure the
  family actually describes?""")
