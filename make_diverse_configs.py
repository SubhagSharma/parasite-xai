"""
make_diverse_configs.py — the five-arm prototype diversity ablation.

Each arm adds ONE mechanism to the previous, so the contribution of each is isolated.
Arm A is the control: `protopnet_diverse` with every weight at zero, which trains
identically to `protopnet`. If A does not reproduce the baseline, the wiring is wrong
and nothing downstream is interpretable.

  A  base        w_orth 0     w_sparse 0     focal F  push std     control
  B  orth        w_orth 0.1   w_sparse 0     focal F  push std     orthogonality alone
  C  orth_sp     w_orth 0.1   w_sparse 0.01  focal F  push std     + channel sparsity
  D  focal       w_orth 0.1   w_sparse 0.01  focal T  push std     + focal similarity
  E  full        w_orth 0.1   w_sparse 0.01  focal T  push diverse + Hungarian push

WHY THIS ORDER
Orthogonality is the mechanism with the clearest theoretical backing here: embeddings
are non-negative after the Sigmoid add_on, so cos = 0 forces disjoint channel support.
Sparsity sharpens that from "disjoint" to "small and disjoint". Focal similarity changes
the forward pass and so cannot be reverted at eval time — it is added late deliberately,
after the cheap changes have been measured. Diverse push touches only the projection
step and is last because it is the most likely to interact badly with the others.

WEIGHTS
w_orth 0.1 and w_sparse 0.01 are starting points, not tuned values. The loss is
    ce + 0.8*cluster - 0.08*sep + 1e-4*l1 + w_orth*orth + w_sparse*sparse
Orthogonality is bounded in [0,1] per pair, so 0.1 puts it an order of magnitude below
the cluster term — enough to shape the solution, not enough to dominate classification.
If arm B shows no movement in `cos`, raise to 0.5 before concluding it does not work.

    python make_diverse_configs.py
    python make_diverse_configs.py --seeds 1337,2337    # both seeds, 10 arms
"""
import argparse
import copy
import os

import yaml

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "generated")

ARMS = [
    ("base",    dict(w_orth=0.0, w_sparse=0.00, focal=False, diverse_push=False)),
    ("orth",    dict(w_orth=0.1, w_sparse=0.00, focal=False, diverse_push=False)),
    ("orth_sp", dict(w_orth=0.1, w_sparse=0.01, focal=False, diverse_push=False)),
    ("focal",   dict(w_orth=0.1, w_sparse=0.01, focal=True,  diverse_push=False)),
    ("full",    dict(w_orth=0.1, w_sparse=0.01, focal=True,  diverse_push=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="roi477_protopnet_120ep")
    ap.add_argument("--seeds", default="1337")
    ap.add_argument("--w-orth", type=float, default=0.1)
    ap.add_argument("--w-sparse", type=float, default=0.01)
    a = ap.parse_args()

    with open(os.path.join(GEN, f"{a.src}.yaml")) as f:
        base = yaml.safe_load(f)

    made = []
    for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
        tag = "" if seed == 1337 else f"_s{seed}"
        for arm, opts in ARMS:
            c = copy.deepcopy(base)
            c["seed"] = seed
            c["model"]["kind"] = "protopnet_diverse"
            p = copy.deepcopy(c["model"].get("protopnet", {}))
            p.update(opts)
            if opts["w_orth"] > 0:
                p["w_orth"] = a.w_orth
            if opts["w_sparse"] > 0:
                p["w_sparse"] = a.w_sparse
            c["model"]["protopnet_diverse"] = p
            name = f"roi477_div_{arm}{tag}_120ep"
            c["output_dir"] = f"./runs/{name}"
            with open(os.path.join(GEN, f"{name}.yaml"), "w") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            made.append(name)
            print(f"  {name:<34} orth={p.get('w_orth',0):<5} "
                  f"sparse={p.get('w_sparse',0):<6} focal={p['focal']:<6} "
                  f"push={'diverse' if p['diverse_push'] else 'std'}")

    print(f"\n{len(made)} configs written from {a.src}.")
    print("Arm 'base' must reproduce the protopnet baseline; if it does not, the "
          "wiring is wrong and nothing else is interpretable.")


if __name__ == "__main__":
    main()
