from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import h5py


def discover(h5_path):
    records = []
    with h5py.File(h5_path, "r") as f:
        for split in f.keys():
            split_grp = f[split]
            if not isinstance(split_grp, h5py.Group):
                continue
            for formula, formula_grp in split_grp.items():
                if not isinstance(formula_grp, h5py.Group):
                    continue
                for rxn_id, rxn_grp in formula_grp.items():
                    if not isinstance(rxn_grp, h5py.Group):
                        continue
                    if "positions" not in rxn_grp:
                        continue
                    records.append({
                        "split": split,
                        "formula": formula,
                        "family": formula,
                        "rxn_id": rxn_id,
                        "n_frames": int(rxn_grp["positions"].shape[0]),
                    })
    return records


def take_records(records, allowed_roots, used_families, n_target, rng):
    fam_to_records = defaultdict(list)
    for r in records:
        if r["split"] in allowed_roots and r["family"] not in used_families:
            fam_to_records[r["family"]].append(r)

    families = list(fam_to_records.keys())
    rng.shuffle(families)

    selected = []
    selected_fams = set()

    for fam in families:
        rs = fam_to_records[fam]
        rng.shuffle(rs)
        for r in rs:
            selected.append(r)
            selected_fams.add(fam)
            if len(selected) >= n_target:
                return selected, selected_fams

    return selected, selected_fams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--out", default="splits_pilot.json")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--n-val", type=int, default=15)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--n-ood", type=int, default=15)
    args = p.parse_args()

    rng = random.Random(args.seed)
    records = discover(args.h5)

    bare_counts = Counter(r["rxn_id"] for r in records)
    records = [r for r in records if bare_counts[r["rxn_id"]] == 1 and r["n_frames"] > 0]

    print(f"[pilot] usable globally unique reactions: {len(records)}")

    used_families = set()

    train, fams = take_records(records, {"train", "data"}, used_families, args.n_train, rng)
    used_families |= fams

    val, fams = take_records(records, {"val"}, used_families, args.n_val, rng)
    used_families |= fams

    test, fams = take_records(records, {"test"}, used_families, args.n_test, rng)
    used_families |= fams

    ood, fams = take_records(records, {"test"}, used_families, args.n_ood, rng)
    used_families |= fams

    def ids(rs):
        return sorted(r["rxn_id"] for r in rs)

    def frames(rs):
        return sum(r["n_frames"] for r in rs)

    split = {
        "train_id": ids(train),
        "val_id": ids(val),
        "test_id_same_family": ids(test),
        "test_ood_family": ids(ood),
        "frozen_blind_test": [],
        "family_split_map": {},
        "frame_counts": {
            "train": frames(train),
            "val": frames(val),
            "test_id": frames(test),
            "test_ood": frames(ood),
        },
        "reaction_counts": {
            "train": len(train),
            "val": len(val),
            "test_id": len(test),
            "test_ood": len(ood),
        },
        "generation_seed": args.seed,
        "notes": [
            "Pilot split only.",
            "Bare rxn_id values are globally unique to avoid current train/eval ambiguity.",
            "Families are disjoint across train/val/test/ood."
        ],
    }

    for r in train:
        split["family_split_map"][r["family"]] = "train"
    for r in val:
        split["family_split_map"][r["family"]] = "val"
    for r in test:
        split["family_split_map"][r["family"]] = "test_id"
    for r in ood:
        split["family_split_map"][r["family"]] = "test_ood"

    all_ids = split["train_id"] + split["val_id"] + split["test_id_same_family"] + split["test_ood_family"]
    no_duplicates = len(all_ids) == len(set(all_ids))

    train_fams = {r["family"] for r in train}
    val_fams = {r["family"] for r in val}
    test_fams = {r["family"] for r in test}
    ood_fams = {r["family"] for r in ood}

    leakage_free = (
        not (train_fams & val_fams)
        and not (train_fams & test_fams)
        and not (train_fams & ood_fams)
        and not (test_fams & ood_fams)
    )

    split["verification"] = {
        "no_duplicate_rxns": no_duplicates,
        "leakage_free": leakage_free,
        "train_val_overlap": sorted(train_fams & val_fams),
        "train_test_overlap": sorted(train_fams & test_fams),
        "train_ood_overlap": sorted(train_fams & ood_fams),
        "test_ood_overlap": sorted(test_fams & ood_fams),
        "PASS": bool(no_duplicates and leakage_free),
    }

    Path(args.out).write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(f"[pilot] wrote {args.out}")
    print("[pilot] reaction_counts:", split["reaction_counts"])
    print("[pilot] frame_counts:", split["frame_counts"])
    print("[pilot] verification:", split["verification"])


if __name__ == "__main__":
    main()