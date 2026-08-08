"""
validate_concepts.py — is this concept table usable as a CBM bottleneck?

A hard bottleneck makes the label a function of the concepts ONLY. If two species map
to the same concept vector, the model provably cannot separate them: best case is a
coin flip on that pair, capping accuracy near 0.909 for one collision out of 11
classes. This checks that and three other things before a single epoch is trained.

CHECKS
  1. INJECTIVITY   — do all 11 species have distinct encoded vectors? Hard blocker.
  2. DIMENSION     — one-hot encoded width, which must equal model.cbm.num_concepts.
  3. BALANCE       — concepts positive for only 1 of 11 species are nearly unlearnable
                     under BCE without class weighting, and are also the ones that
                     carry the most discriminative information. Flagged, not fatal.
  4. SCALE AGREEMENT — cross-checks size_band against the empirically fitted relative
                     egg sizes from probe_scale_decomposition.py. A textbook band that
                     contradicts the measurement on your own data is worth knowing
                     about before it becomes a supervision target.

    python -u validate_concepts.py --csv concepts_v2.csv
    python -u validate_concepts.py --csv concepts_v2.csv --classes-from ../Data/chula_roi2_w477
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

# Relative egg sizes fitted by probe_scale_decomposition.py on chula_roi2, acquisition
# scale removed. Smallest = 1.00. Anchoring at Opisthorchis ~28 um reproduced published
# dimensions for 9 of 11 species, so this is a measurement, not a guess.
MEASURED_RELATIVE = {
    "Opisthorchis viverrini": 1.00, "Opisthorchis viverrine": 1.00,
    "Taenia spp. egg": 1.46, "Capillaria philippinensis": 1.51,
    "Hymenolepis nana": 1.71, "Trichuris trichiura": 1.97,
    "Enterobius vermicularis": 2.04, "Hookworm egg": 2.38,
    "Ascaris lumbricoides": 2.43, "Hymenolepis diminuta": 2.55,
    "Paragonimus spp.": 2.77, "Paragonimus spp": 2.77,
    "Fasciolopsis buski": 4.62,
}
BAND_ORDER = {"very_small": 0, "small": 1, "medium": 2, "large": 3, "very_large": 4}


def encode(rows, key="species"):
    cols = [c for c in rows[0] if c != key]
    enc = {}
    for c in cols:
        vals = sorted({r[c] for r in rows})
        enc[c] = ("binary", 1, vals) if set(vals) <= {"0", "1"} \
            else ("onehot", len(vals), vals)

    def vec(r):
        out = []
        for c in cols:
            kind, _, vals = enc[c]
            if kind == "binary":
                out.append(int(r[c]))
            else:
                out += [1 if r[c] == v else 0 for v in vals]
        return tuple(out)

    names = []
    for c in cols:
        kind, _, vals = enc[c]
        names += [c] if kind == "binary" else [f"{c}={v}" for v in vals]
    return {r[key]: vec(r) for r in rows}, names, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--classes-from", default=None,
                    help="dataset root; checks the species strings match folder names")
    ap.add_argument("--num-concepts", type=int, default=16,
                    help="value in the config, to compare against the encoded width")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    V, names, enc = encode(rows)
    dim = len(names)
    print(f"{len(rows)} species, {len(enc)} raw columns -> {dim} encoded concepts\n")

    print(f"{'column':<24}{'type':<9}{'dims':>5}  values")
    for c, (kind, d, vals) in enc.items():
        print(f"  {c:<22}{kind:<9}{d:>5}  {vals}")

    ok = True

    # 1. injectivity
    print("\n1. INJECTIVITY")
    hd = sorted((sum(x != y for x, y in zip(V[p], V[q])), p, q)
                for p, q in itertools.combinations(V, 2))
    if hd[0][0] == 0:
        ok = False
        print("   FAIL — species share an identical concept vector. A hard bottleneck")
        print("   cannot separate them; accuracy is capped at chance on each pair.")
        for d, p, q in hd:
            if d:
                break
            print(f"     COLLISION: {p}  ==  {q}")
    else:
        print(f"   PASS — all distinct, minimum Hamming distance {hd[0][0]}")
    print("   closest pairs:")
    for d, p, q in hd[:3]:
        print(f"     {d}  {p}  vs  {q}")

    # 2. dimension
    print(f"\n2. DIMENSION")
    if dim == a.num_concepts:
        print(f"   PASS — {dim} matches model.cbm.num_concepts")
    else:
        ok = False
        print(f"   MISMATCH — encoded {dim}, config says {a.num_concepts}.")
        print(f"   Set model.cbm.num_concepts: {dim} in every cbm config.")

    # 3. balance
    print("\n3. BALANCE  (positives out of 11)")
    rare = []
    for i, nm in enumerate(names):
        n = sum(v[i] for v in V.values())
        flag = ""
        if n <= 1:
            flag = "** RARE"
            rare.append(nm)
        elif n >= len(V) - 1:
            flag = "** NEAR-CONSTANT"
            rare.append(nm)
        print(f"   {nm:<32}{n:>3}/{len(V)}  {flag}")
    if rare:
        print(f"\n   {len(rare)} concept(s) are positive (or negative) for a single")
        print("   species. Under unweighted BCE the model can score well by predicting")
        print("   the majority everywhere, so per-concept accuracy will look good while")
        print("   these carry no signal. Use pos_weight in the concept loss, and report")
        print("   per-concept balanced accuracy rather than raw accuracy.")

    # 4. scale agreement
    if "size_band" in enc:
        print("\n4. SIZE_BAND vs MEASURED EGG SIZE")
        band = {r["species"]: r["size_band"] for r in rows}
        known = [(s, MEASURED_RELATIVE[s]) for s in band if s in MEASURED_RELATIVE]
        known.sort(key=lambda t: t[1])
        missing = [s for s in band if s not in MEASURED_RELATIVE]
        prev = None
        bad = 0
        print(f"   {'species':<28}{'measured':>9}  band")
        for s, m in known:
            note = ""
            if prev and BAND_ORDER.get(band[s], 9) < BAND_ORDER.get(band[prev], 9):
                note = f"** INVERTED vs {prev}"
                bad += 1
            print(f"   {s:<28}{m:>9.2f}  {band[s]:<12}{note}")
            prev = s
        if missing:
            print(f"   (no measurement for: {missing})")
        if bad:
            print(f"\n   {bad} inversion(s): the textbook band contradicts the measured")
            print("   size on THIS dataset. Supervising a concept against a label your")
            print("   own data disagrees with will show up as low per-concept accuracy")
            print("   that looks like a model failure but is a label error. Either")
            print("   re-derive size_band from MEASURED_RELATIVE, or justify the")
            print("   discrepancy (e.g. the band refers to a dimension not captured by")
            print("   the annotation box).")
        else:
            print("\n   PASS — band ordering is consistent with the measurement.")

    # 5. folder names
    if a.classes_from:
        print("\n5. SPECIES STRINGS vs DATASET FOLDERS")
        if not os.path.isdir(a.classes_from):
            print(f"   SKIP — {a.classes_from} not found")
        else:
            folders = {d for d in os.listdir(a.classes_from)
                       if os.path.isdir(os.path.join(a.classes_from, d))}
            csv_names = set(band if False else {r["species"] for r in rows})
            only_csv = csv_names - folders
            only_dir = folders - csv_names
            if not only_csv and not only_dir:
                print(f"   PASS — all {len(folders)} names match exactly")
            else:
                ok = False
                print("   MISMATCH — a string join will silently drop rows.")
                for s in sorted(only_csv):
                    print(f"     in CSV only: {s!r}")
                for s in sorted(only_dir):
                    print(f"     in dataset only: {s!r}")
                print("   Add an explicit alias map rather than relying on the join.")

    print("\n" + "=" * 60)
    print("USABLE AS A BOTTLENECK" if ok else "NOT USABLE — fix the failures above")
    print("=" * 60)
    print("\nRemember: these are CLASS-LEVEL concepts, so c = M[y] is a deterministic")
    print("function of the label, unlike Koh et al.'s per-image CUB annotations. Name it")
    print("as class-level supervision in the writeup. The measurement it enables is still")
    print("real: concept accuracy BELOW class accuracy means the bottleneck leaks -- the")
    print("model is getting the diagnosis right without getting the morphology right.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()