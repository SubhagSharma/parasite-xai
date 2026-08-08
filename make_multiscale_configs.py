"""
make_multiscale_configs.py — prototypes at more than one backbone depth.

Four arms. Every one keeps 5 prototypes per class, so the comparison against the
single-scale baseline is like-for-like and any difference is the DEPTH they attach to.

  ms_coarse   stages [3]      5 coarse            control: should reproduce protopnet
  ms_fine     stages [1]      5 fine              all prototypes at stride 8
  ms_2way     stages [1,3]    2 fine + 3 coarse   the proposal
  ms_3way     stages [0,1,3]  1 + 2 + 2           three scales

WHY THE CONTROL MATTERS
`ms_coarse` puts every prototype at the same stage the standard head uses, through the
new code path. If it does not reproduce roi477_protopnet, the multi-scale wiring is
wrong and the other three arms mean nothing. Check it first.

WHAT THE STAGES ARE
mobilevit_xs emits five feature maps. Verify the reduction factors on your build:

    python -c "
    import timm
    m = timm.create_model('mobilevit_xs', pretrained=False, features_only=True)
    print('channels ', m.feature_info.channels())
    print('reduction', m.feature_info.reduction())"

Typical: strides [4, 8, 16, 32, 32], channels [32, 48, 64, 80, 384]. Stage 1 is
stride 8 -- one cell is 8 px, which is the scale of cilia and polar plugs. Stage 3/4 is
stride 32, one cell 32 px, which is whole-egg shape.

If your reduction list differs, pass --stages-fine and --stages-coarse to match.

PREDICTION, RECORDED BEFORE THE RUN
Under the receptive-field model (MECHANISTIC_MODEL.md §3), conc_grad ~ 1/rho. A
stride-8 prototype has an RF a fraction of the object rather than several times it, so
ms_fine should localise ABOVE the ~5.5 ceiling the stride-32 heads hit. If fine and
coarse score the same, the RF model is wrong and §3 needs withdrawing.

    python make_multiscale_configs.py
"""
import argparse
import copy
import os

import yaml

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "generated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="roi477_protopnet_120ep")
    ap.add_argument("--fine", type=int, default=1, help="stage index for fine scale")
    ap.add_argument("--mid", type=int, default=0, help="stage index for the 3-way arm")
    ap.add_argument("--coarse", type=int, default=3, help="stage index for coarse scale")
    ap.add_argument("--seeds", default="1337")
    a = ap.parse_args()

    src = os.path.join(GEN, f"{a.src}.yaml")
    if not os.path.exists(src):
        raise SystemExit(f"source config not found: {src}")
    base = yaml.safe_load(open(src))

    arms = [
        ("ms_coarse", [a.coarse],                 [5]),
        ("ms_fine",   [a.fine],                   [5]),
        ("ms_2way",   [a.fine, a.coarse],         [2, 3]),
        ("ms_3way",   [a.mid, a.fine, a.coarse],  [1, 2, 2]),
    ]

    made = []
    for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
        tag = "" if seed == 1337 else f"_s{seed}"
        for arm, stages, ppcs in arms:
            c = copy.deepcopy(base)
            c["seed"] = seed
            c["model"]["kind"] = "protopnet_ms"
            c["model"]["protopnet_ms"] = {
                "stages": stages,
                "protos_per_class_per_stage": ppcs,
                "proto_dim": base["model"].get("protopnet", {}).get("proto_dim", 128),
                "pip_sparsity": base["model"].get("protopnet", {}).get(
                    "pip_sparsity", True),
            }
            name = f"roi477_{arm}{tag}_120ep"
            c["output_dir"] = f"./runs/{name}"
            with open(os.path.join(GEN, f"{name}.yaml"), "w") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            made.append(name)
            print(f"  {name:<30} stages {str(stages):<12} "
                  f"protos/class {ppcs}  total/class {sum(ppcs)}")

    print(f"\n{len(made)} configs from {a.src}. All keep 5 prototypes per class.")
    print("Check ms_coarse against roi477_protopnet FIRST -- it is the wiring control.")


if __name__ == "__main__":
    main()
