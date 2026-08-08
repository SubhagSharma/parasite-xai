"""make_parts_config.py — config for the spatially-grounded concept-part head.

Derived from roi477_cbm_sup_120ep so the concept table, its path and the bottleneck
flag are identical. The ONLY difference is that each concept is predicted from its own
spatial attention slot instead of from a global average -- which is what makes the
attention map a localisation of that concept.
"""
import yaml

c = yaml.safe_load(open("configs/generated/roi477_cbm_sup_120ep.yaml"))
cb = c["model"].get("cbm", {})
c["model"]["kind"] = "concept_parts"
c["model"]["concept_parts"] = {
    "num_concepts": cb.get("num_concepts", 23),
    "concepts_csv": cb.get("concepts_csv",
                           "../Data/Chula-ParasiteEgg-11/concepts_v3.csv"),
    "slot_dim": 128,
    "w_compact": 0.1,     # a morphological feature is a connected region
    "w_distinct": 0.1,    # only across morphological families, not within
    "w_presence": 0.1,    # every slot active somewhere -- prevents dead slots
    "bottleneck": True,
}
c["output_dir"] = "./runs/roi477_parts_120ep"
yaml.safe_dump(c, open("configs/generated/roi477_parts_120ep.yaml", "w"),
               sort_keys=False)
print(f"wrote roi477_parts_120ep.yaml  "
      f"({c['model']['concept_parts']['num_concepts']} concepts, "
      f"csv {c['model']['concept_parts']['concepts_csv']})")
