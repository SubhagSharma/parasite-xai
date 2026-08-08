"""
make_night6_configs.py — supervised CBM, plus the seed-2337 replication set.

WRITES
  roi477_cbm_sup_120ep          supervised 23-concept bottleneck, NEW output dir so
                                the 16-dim unsupervised checkpoint survives (the two
                                state dicts are not even shape-compatible)
  roi477_{blackbox,protopnet,bcos,cbm_sup}_s2337_120ep
                                identical configs at seed 2337

WHY THE SEED SET
----------------
The technical report marks the head ordering (protopnet < cbm < blackbox < bcos on
egg-masked accuracy) PROVISIONAL, with the note "needs replication at a second seed
before it can anchor a paper section". That ordering is the C1/C3 headline. Four
retrains at ~55 min each plus the occlusion and displaced-control probes is ~4h and
converts the project's main result from provisional to established -- cheaper than a
single faithfulness eval and worth more.

ALSO FIXED
----------
concept_source: labelfree -> labels. The configs say "labelfree" (Oikarinen et al.:
concepts mined from an LLM and grounded with CLIP), but concepts_v3.csv is an explicit
morphology table, so the supervision is `labels`. Nothing in the code reads this key
today; it is metadata that would misdescribe the method in the writeup.

    python make_night6_configs.py
"""
import argparse
import copy
import os

import yaml

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "generated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2337)
    ap.add_argument("--csv", default="../Data/Chula-ParasiteEgg-11/concepts_v3.csv")
    ap.add_argument("--num-concepts", type=int, default=23)
    a = ap.parse_args()

    def load(name):
        with open(os.path.join(GEN, f"{name}.yaml")) as f:
            return yaml.safe_load(f)

    def write(name, cfg):
        cfg["output_dir"] = f"./runs/{name}"
        with open(os.path.join(GEN, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  {name:<34} seed={cfg['seed']:<6} {cfg['model']['kind']:<10} "
              f"{cfg['data']['root']}")

    # 1. supervised CBM at the canonical seed
    c = copy.deepcopy(load("roi477_cbm_120ep"))
    c["model"]["cbm"]["num_concepts"] = a.num_concepts
    c["model"]["cbm"]["concepts_csv"] = a.csv
    c["model"]["cbm"]["concept_source"] = "labels"
    write("roi477_cbm_sup_120ep", c)

    # 2. seed replication, all four heads
    for head, src in (("blackbox", "roi477_blackbox_120ep"),
                      ("protopnet", "roi477_protopnet_120ep"),
                      ("bcos", "roi477_bcos_120ep"),
                      ("cbm_sup", "roi477_cbm_sup_120ep")):
        c = copy.deepcopy(load(src if head != "cbm_sup" else "roi477_cbm_sup_120ep"))
        c["seed"] = a.seed
        write(f"roi477_{head}_s{a.seed}_120ep", c)

    print(f"\n5 configs written. Everything except seed, model.kind, the concept keys")
    print("and output_dir matches the existing 120ep family, so any difference is")
    print("attributable to the head and the seed alone.")


if __name__ == "__main__":
    main()
