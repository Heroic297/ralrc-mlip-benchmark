import argparse
from pathlib import Path
import h5py

ENDPOINTS = ("reactant", "product", "transition_state")

def collect_split(f, split):
    out = {
        "formulas": set(),
        "rxns": set(),
        "endpoint_hashes": {ep: set() for ep in ENDPOINTS},
        "n_reactions": 0,
    }

    if split not in f:
        return out

    for formula, formula_grp in f[split].items():
        out["formulas"].add(formula)

        for rxn_id, rxn_grp in formula_grp.items():
            out["rxns"].add((formula, rxn_id))
            out["n_reactions"] += 1

            for ep in ENDPOINTS:
                if ep in rxn_grp and "hash" in rxn_grp[ep]:
                    try:
                        out["endpoint_hashes"][ep].add(int(rxn_grp[ep]["hash"][()]))
                    except Exception:
                        pass

    return out

def pct(n, d):
    return 100.0 * n / d if d else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    args = parser.parse_args()

    path = Path(args.h5)

    with h5py.File(path, "r") as f:
        splits = ["train", "val", "test"]
        data = {s: collect_split(f, s) for s in splits}

    print("=== Split sizes ===")
    for s in splits:
        print(f"{s:5s}: formulas={len(data[s]['formulas']):4d} reactions={data[s]['n_reactions']:6d}")

    print("\n=== Formula overlap ===")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = data[a]["formulas"] & data[b]["formulas"]
        union = data[a]["formulas"] | data[b]["formulas"]
        print(f"{a:5s} vs {b:5s}: overlap={len(inter):4d} union={len(union):4d} jaccard={pct(len(inter), len(union)):.2f}%")
        if inter:
            print("  examples:", sorted(list(inter))[:20])

    print("\n=== Reaction ID overlap ===")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = data[a]["rxns"] & data[b]["rxns"]
        union = data[a]["rxns"] | data[b]["rxns"]
        print(f"{a:5s} vs {b:5s}: overlap={len(inter):4d} union={len(union):5d} jaccard={pct(len(inter), len(union)):.4f}%")

    print("\n=== Endpoint hash overlap ===")
    for ep in ENDPOINTS:
        print(f"\nendpoint = {ep}")
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            inter = data[a]["endpoint_hashes"][ep] & data[b]["endpoint_hashes"][ep]
            union = data[a]["endpoint_hashes"][ep] | data[b]["endpoint_hashes"][ep]
            print(f"{a:5s} vs {b:5s}: overlap={len(inter):4d} union={len(union):5d} jaccard={pct(len(inter), len(union)):.4f}%")

if __name__ == "__main__":
    main()
