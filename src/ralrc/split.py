"""Leakage-safe split generator for Transition1x by reaction family.

Strategy
--------
1. Parse the HDF5 file to discover all (formula, rxn_id) pairs.
2. Derive a "reaction family" from the molecular formula (element composition).
   This groups reactions by stoichiometric family, not geometry, avoiding
   geometry-level leakage between splits.
3. Shuffle families with a fixed seed and assign:
   - 70% of families -> train
   - 15% of families -> val
   - 10% of families -> test_id_same_family  (seen families, unseen reactions)
   -  5% of families -> test_ood_family      (held-out families entirely)
4. Writes splits.json with reaction ID lists for each split.
5. Prints a leakage verification report confirming zero family overlap
   between train and test_ood.

Usage:
    python -m ralrc.split --h5 data/transition1x.h5 --out splits.json \\
        --seed 17 --ood-fraction 0.05

Verification:
    python -m ralrc.split --h5 data/transition1x.h5 --verify splits.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 discovery
# ---------------------------------------------------------------------------

def discover_reactions(h5_path: str | Path) -> list[dict]:
    """
    Walk the Transition1x HDF5 and return a list of reaction descriptor dicts:
      {
        'rxn_id':  str,
        'formula': str,   # molecular formula string used as family key
        'family':  str,   # same as formula for stoichiometric grouping
        'split':   str,   # original HDF5 split label
        'n_frames': int,
      }
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    records = []
    with h5py.File(h5_path, "r") as f:
        for split in f.keys():
            split_grp = f[split]
            if not isinstance(split_grp, h5py.Group):
                continue
            for formula, frml_grp in split_grp.items():
                if not isinstance(frml_grp, h5py.Group):
                    continue
                for rxn_id, rxn_grp in frml_grp.items():
                    if not isinstance(rxn_grp, h5py.Group):
                        continue
                    n_frames = 0
                    if "positions" in rxn_grp:
                        n_frames = rxn_grp["positions"].shape[0]
                    records.append({
                        "rxn_id":   rxn_id,
                        "formula":  formula,
                        "family":   formula,   # stoichiometric family
                        "split":    split,
                        "n_frames": n_frames,
                    })
    return records


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def build_family_split(
    records: list[dict],
    seed: int = 17,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    test_id_frac: float = 0.10,
    ood_frac:   float = 0.05,
) -> dict:
    """
    Assign every reaction to a split by shuffling families deterministically.

    Returns a dict:
      {
        'train_id':           [rxn_id, ...],
        'val_id':             [rxn_id, ...],
        'test_id_same_family': [rxn_id, ...],
        'test_ood_family':    [rxn_id, ...],
        'frozen_blind_test':  [],
        'family_split_map':   {family: split_label},
        'frame_counts':       {split_label: int},
        'reaction_counts':    {split_label: int},
      }
    """
    assert abs(train_frac + val_frac + test_id_frac + ood_frac - 1.0) < 1e-9, \
        "Fractions must sum to 1.0"

    # Group reactions by family
    family_to_rxns: dict[str, list[str]] = defaultdict(list)
    rxn_to_info: dict[str, dict] = {}
    for r in records:
        family_to_rxns[r["family"]].append(r["rxn_id"])
        rxn_to_info[r["rxn_id"]] = r

    families = sorted(family_to_rxns.keys())
    rng = random.Random(seed)
    rng.shuffle(families)

    n = len(families)
    n_ood  = max(1, int(round(ood_frac   * n)))
    n_test = max(1, int(round(test_id_frac * n)))
    n_val  = max(1, int(round(val_frac   * n)))
    n_train = n - n_ood - n_test - n_val

    # Assign families in order: train | val | test_id | ood
    ood_families    = set(families[n_train + n_val + n_test:])
    test_id_families = set(families[n_train + n_val: n_train + n_val + n_test])
    val_families    = set(families[n_train: n_train + n_val])
    train_families  = set(families[:n_train])

    split_label: dict[str, str] = {}
    for fam in train_families:  split_label[fam] = "train"
    for fam in val_families:    split_label[fam] = "val"
    for fam in test_id_families: split_label[fam] = "test_id"
    for fam in ood_families:    split_label[fam] = "test_ood"

    train_ids, val_ids, test_ids, ood_ids = [], [], [], []
    for fam, rxns in family_to_rxns.items():
        label = split_label[fam]
        if label == "train":    train_ids.extend(rxns)
        elif label == "val":    val_ids.extend(rxns)
        elif label == "test_id": test_ids.extend(rxns)
        elif label == "test_ood": ood_ids.extend(rxns)

    def _count_frames(rxn_ids):
        return sum(rxn_to_info[r]["n_frames"] for r in rxn_ids)

    result = {
        "train_id":            sorted(train_ids),
        "val_id":              sorted(val_ids),
        "test_id_same_family": sorted(test_ids),
        "test_ood_family":     sorted(ood_ids),
        "frozen_blind_test":   [],
        "family_split_map":    split_label,
        "frame_counts": {
            "train":   _count_frames(train_ids),
            "val":     _count_frames(val_ids),
            "test_id": _count_frames(test_ids),
            "test_ood": _count_frames(ood_ids),
        },
        "reaction_counts": {
            "train":   len(train_ids),
            "val":     len(val_ids),
            "test_id": len(test_ids),
            "test_ood": len(ood_ids),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_leakage(splits: dict, records: list[dict]) -> dict:
    """
    Check that:
    1. No family appears in both train and test_ood (critical)
    2. Every rxn_id appears in exactly one split
    3. Frame counts are approximately balanced (train >= others)

    Returns a report dict.
    """
    rxn_to_info = {r["rxn_id"]: r for r in records}

    # Build family sets per split
    def families_of(rxn_ids):
        return set(rxn_to_info[r]["family"] for r in rxn_ids if r in rxn_to_info)

    train_fams = families_of(splits["train_id"])
    val_fams   = families_of(splits["val_id"])
    test_id_fams = families_of(splits["test_id_same_family"])
    ood_fams   = families_of(splits["test_ood_family"])

    # Critical: no overlap between train and OOD
    train_ood_overlap   = train_fams & ood_fams
    train_testid_overlap = train_fams & test_id_fams  # expected: can overlap

    # Check each rxn_id is in exactly one split
    all_splits = (
        splits["train_id"]
        + splits["val_id"]
        + splits["test_id_same_family"]
        + splits["test_ood_family"]
    )
    duplicates = [r for r in all_splits if all_splits.count(r) > 1]

    # Frame balance check
    fc = splits.get("frame_counts", {})
    frame_balance_ok = fc.get("train", 0) >= fc.get("val", 0)

    report = {
        "leakage_free":         len(train_ood_overlap) == 0,
        "train_ood_overlap":    sorted(train_ood_overlap),
        "no_duplicate_rxns":    len(duplicates) == 0,
        "duplicate_rxn_ids":    list(set(duplicates)),
        "frame_balance_ok":     frame_balance_ok,
        "frame_counts":         fc,
        "reaction_counts":      splits.get("reaction_counts", {}),
        "n_train_families":     len(train_fams),
        "n_val_families":       len(val_fams),
        "n_test_id_families":   len(test_id_fams),
        "n_ood_families":       len(ood_fams),
        "PASS": (
            len(train_ood_overlap) == 0
            and len(duplicates) == 0
        ),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build leakage-safe family splits for Transition1x.")
    p.add_argument("--h5",           required=True, help="Path to Transition1x.h5")
    p.add_argument("--out",          default="splits.json")
    p.add_argument("--seed",         type=int, default=17)
    p.add_argument("--ood-fraction", type=float, default=0.05)
    p.add_argument("--verify",       metavar="SPLITS_JSON",
                   help="Verify an existing splits.json rather than generating a new one")
    a = p.parse_args()

    print(f"[split] Discovering reactions in {a.h5}...")
    records = discover_reactions(a.h5)
    print(f"[split] Found {len(records)} reactions across "
          f"{len(set(r['family'] for r in records))} families.")

    # --- Verify mode ---
    if a.verify:
        with open(a.verify) as f:
            splits = json.load(f)
        report = verify_leakage(splits, records)
        print(json.dumps(report, indent=2))
        if report["PASS"]:
            print("[split] VERIFICATION PASSED — no leakage detected.")
        else:
            print("[split] VERIFICATION FAILED — see report above.")
        return

    # --- Generate mode ---
    # Check frame imbalance before splitting
    total_frames = sum(r["n_frames"] for r in records)
    print(f"[split] Total frames in dataset: {total_frames}")

    splits = build_family_split(
        records,
        seed=a.seed,
        ood_frac=a.ood_fraction,
        train_frac=0.70,
        val_frac=0.15,
        test_id_frac=1.0 - 0.70 - 0.15 - a.ood_fraction,
    )

    # Verify immediately
    report = verify_leakage(splits, records)

    output = {**splits, "verification": report, "generation_seed": a.seed}
    with open(a.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[split] Wrote {a.out}")
    print(f"[split] Reaction counts: {splits['reaction_counts']}")
    print(f"[split] Frame counts:    {splits['frame_counts']}")

    if report["PASS"]:
        print("[split] Leakage check PASSED.")
    else:
        print("[split] WARNING: Leakage check FAILED — see splits.json for details.")

    # Warn if train has fewer frames than val/test (the known pilot issue)
    fc = splits["frame_counts"]
    if fc["train"] < fc["val"]:
        print(
            f"[split] WARNING: train has fewer frames ({fc['train']}) than val "
            f"({fc['val']}). This indicates reactions in the train split have "
            f"shorter IRC paths on average. Consider:\n"
            f"  (a) Oversampling train reactions (duplicate short IRCs), or\n"
            f"  (b) Downsampling val/test to match train frame count, or\n"
            f"  (c) Accepting the imbalance if IRC path lengths are genuinely shorter.\n"
            f"  Run: python -m ralrc.split_diagnostics --h5 {a.h5} --splits {a.out}"
        )


if __name__ == "__main__":
    main()
