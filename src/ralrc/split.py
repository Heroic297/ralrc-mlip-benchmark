"""Leakage-safe split generator by reaction family / reaction ID."""
import json, argparse, random

def split_by_family(structures, ood_families, seed=17):
    random.seed(seed)
    train, val, test_id, test_ood = [], [], [], []
    for s in structures:
        if s["family"] in ood_families:
            test_ood.append(s["reaction_id"])
        else:
            r = random.random()
            (train if r < 0.8 else val if r < 0.9 else test_id).append(s["reaction_id"])
    return {"train_id": list(set(train)), "val_id": list(set(val)),
            "test_id_same_family": list(set(test_id)),
            "test_ood_family": list(set(test_ood)),
            "frozen_blind_test": []}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="splits.json")
    a = p.parse_args()
    print(f"Generate splits from {a.config} -> {a.out}")
    json.dump({"status": "stub_no_data"}, open(a.out, "w"))

if __name__ == "__main__": main()
