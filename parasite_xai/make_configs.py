"""
make_configs.py — generate every experiment config from configs/default.yaml.

Run ONCE after `pip install -r requirements.txt` and after you have laid out
data/chula. It writes configs/generated/*.yaml — one file per experiment in the
ablation ladder (A1-A7) plus the black-box reference and the three interpretable
heads. The daily playbook launches these by name; you never hand-edit YAML.

    python make_configs.py

Every generated config sets a distinct output_dir under runs/ so nothing
overwrites anything.
"""
import copy, os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "configs", "default.yaml")
OUT  = os.path.join(HERE, "configs", "generated")


def load_base():
    with open(BASE) as f:
        return yaml.safe_load(f)


def dump(cfg, name):
    os.makedirs(OUT, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = f"./runs/{name}"
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {os.path.relpath(path, HERE)}  ->  output_dir=runs/{name}")


def main():
    base = load_base()
    n = 0

    # --- REFERENCE: heavy black box (accuracy ceiling + SHAP teacher) ---
    c = copy.deepcopy(base)
    c["backbone"]["name"] = "convnext_tiny"
    c["model"]["kind"] = "blackbox"
    dump(c, "ref_blackbox_convnext"); n += 1

    # --- A2: three interpretable heads on the light backbone ---
    for head in ["protopnet", "cbm", "bcos"]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "mobilevit_xs"
        c["model"]["kind"] = head
        dump(c, f"A2_{head}_mobilevit"); n += 1

    # --- A1: backbone footprint sweep (uses protopnet head; swap after you pick the winner) ---
    for bb in ["mobilevit_xs", "efficientnet_lite0", "ghostnet_100", "convnext_tiny"]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = bb
        c["model"]["kind"] = "protopnet"
        dump(c, f"A1_{bb}"); n += 1

    # --- A3: prototype / concept sparsity sweep ---
    for k in [1, 3, 5, 10]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "mobilevit_xs"
        c["model"]["kind"] = "protopnet"
        c["model"]["protopnet"]["num_prototypes_per_class"] = k
        dump(c, f"A3_proto{k}"); n += 1
    for nc in [8, 16, 32]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "mobilevit_xs"
        c["model"]["kind"] = "cbm"
        c["model"]["cbm"]["num_concepts"] = nc
        dump(c, f"A3_concepts{nc}"); n += 1

    # --- A4: amortized-explainer fidelity vs training budget ---
    for ep in [10, 30, 60]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "convnext_tiny"
        c["model"]["kind"] = "blackbox"
        c["explain"]["amortized"]["enabled"] = True
        c["explain"]["amortized"]["train_epochs"] = ep
        c["explain"]["amortized"]["teacher_ckpt"] = "./runs/ref_blackbox_convnext/best.pt"
        dump(c, f"A4_amortized_ep{ep}"); n += 1

    # --- A5: explanation distillation from the heavy teacher (explicit teacher ckpt) ---
    c = copy.deepcopy(base)
    c["backbone"]["name"] = "convnext_tiny"
    c["model"]["kind"] = "blackbox"
    c["explain"]["amortized"]["enabled"] = True
    c["explain"]["amortized"]["teacher_ckpt"] = "./runs/ref_blackbox_convnext/best.pt"
    dump(c, "A5_distilled_amortized"); n += 1

    # --- A6: trust layer ON vs OFF (conformal abstention) ---
    for on in [True, False]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "mobilevit_xs"
        c["model"]["kind"] = "protopnet"
        c["trust"]["conformal"] = on
        dump(c, f"A6_conformal_{'on' if on else 'off'}"); n += 1

    # --- A7: INT8 quantization vs faithfulness ---
    for q in [False, True]:
        c = copy.deepcopy(base)
        c["backbone"]["name"] = "mobilevit_xs"
        c["model"]["kind"] = "protopnet"
        c["eval"]["quantize_int8"] = q
        dump(c, f"A7_int8_{'on' if q else 'off'}"); n += 1

    print(f"\n{n} configs written to configs/generated/")
    print("Launch any with:  python -m pxai.train --config configs/generated/<name>.yaml")


if __name__ == "__main__":
    main()
