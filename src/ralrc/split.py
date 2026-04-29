"""Leakage-safe split generator for Transition1x by reaction family.

Strategy
--------
1. Parse the HDF5 file to discover all (split, formula, rxn_id) triples.
2. Build a COMPOUND KEY  split::formula::rxn_id  that is globally unique.
3. Derive a "reaction family" from the molecular formula (element composition).
4. Shuffle families with a fixed seed and assign:
   - 70% of families -> train
   - 15% of families -> val
   - 10% of families -> test_id_same_family  (seen families, unseen reactions)
   -  5% of families -> test_ood_family      (held-out families entirely)
5. Writes splits.json whose lists contain COMPOUND KEYS, not bare rxn_ids.
6. Prints a leakage verification report.

Compound key format:  "<hdf5_split>::<formula>::<rxn_id>"
  e.g.  "train::C2H5NO::r0"

Train/eval code must build keys as:
    key = f"{entry[0]}::{entry[1]}::{entry[2]}"
where entry = (split, formula, rxn_id, frame_idx, endpoint).

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
    Walk the Transition1x HDF5 and return one record per (split, formula, rxn_id):
      {
        'compound_key': '<split>::<formula>::<rxn_id>',
        'rxn_id':  str,
        'formula': str,
        'family':  str,   # same as formula for stoichiometric grouping
        'split':   str,   # original HDF5 split label
        'n_frames': int,
      }
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    # Only index the canonical splits — "data" is an aggregate group in Transition1x
    # that duplicates train/val/test reactions with the same rxn_ids.  Including it
    # generates "data::<formula>::<rxn_id>" compound keys that the dataset loader never
    # produces, so those keys are silently unmatchable during training/eval.
    CANONICAL_SPLITS = {"train", "val", "test"}

    records = []
    with h5py.File(h5_path, "r") as f:
        for split in f.keys():
            if split not in CANONICAL_SPLITS:
                continue
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
                    compound_key = f"{split}::{formula}::{rxn_id}"
                    records.append({
                        "compound_key": compound_key,
                        "rxn_id":   rxn_id,
                        "formula":  formula,
                        "family":   formula,
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
    max_reactions_per_split: Optional[int] = None,
) -> dict:
    """
    Assign every reaction compound key to a split by shuffling families.

    Returns a dict whose ID lists contain COMPOUND KEYS.
    """
    assert abs(train_frac + val_frac + test_id_frac + ood_frac - 1.0) < 1e-9, \
        "Fractions must sum to 1.0"

    # Group compound keys by family
    family_to_keys: dict[str, list[str]] = defaultdict(list)
    key_to_info: dict[str, dict] = {}
    for r in records:
        family_to_keys[r["family"]].append(r["compound_key"])
        key_to_info[r["compound_key"]] = r

    families = sorted(family_to_keys.keys())
    rng = random.Random(seed)
    rng.shuffle(families)

    n = len(families)
    n_ood  = max(1, int(round(ood_frac   * n)))
    n_test = max(1, int(round(test_id_frac * n)))
    n_val  = max(1, int(round(val_frac   * n)))
    n_train = n - n_ood - n_test - n_val

    ood_families     = set(families[n_train + n_val + n_test:])
    test_id_families = set(families[n_train + n_val: n_train + n_val + n_test])
    val_families     = set(families[n_train: n_train + n_val])
    train_families   = set(families[:n_train])

    split_label: dict[str, str] = {}
    for fam in train_families:   split_label[fam] = "train"
    for fam in val_families:     split_label[fam] = "val"
    for fam in test_id_families: split_label[fam] = "test_id"
    for fam in ood_families:     split_label[fam] = "test_ood"

    train_keys, val_keys, test_keys, ood_keys = [], [], [], []
    for fam, keys in family_to_keys.items():
        label = split_label[fam]
        if label == "train":    train_keys.extend(keys)
        elif label == "val":    val_keys.extend(keys)
        elif label == "test_id": test_keys.extend(keys)
        elif label == "test_ood": ood_keys.extend(keys)

    # Optional pilot subsampling
    if max_reactions_per_split is not None:
        def _cap(lst, n):
            rng2 = random.Random(seed + 1)
            shuffled = list(lst)
            rng2.shuffle(shuffled)
            return shuffled[:n]
        train_keys = _cap(train_keys, max_reactions_per_split)
        val_keys   = _cap(val_keys,   max(1, max_reactions_per_split // 4))
        test_keys  = _cap(test_keys,  max(1, max_reactions_per_split // 4))
        ood_keys   = _cap(ood_keys,   max(1, max_reactions_per_split // 8))

    def _count_frames(keys):
        return sum(key_to_info[k]["n_frames"] for k in keys)

    result = {
        "train_id":            sorted(train_keys),
        "val_id":              sorted(val_keys),
        "test_id_same_family": sorted(test_keys),
        "test_ood_family":     sorted(ood_keys),
        "frozen_blind_test":   [],
        "family_split_map":    split_label,
        "frame_counts": {
            "train":    _count_frames(train_keys),
            "val":      _count_frames(val_keys),
            "test_id":  _count_frames(test_keys),
            "test_ood": _count_frames(ood_keys),
        },
        "reaction_counts": {
            "train":    len(train_keys),
            "val":      len(val_keys),
            "test_id":  len(test_keys),
            "test_ood": len(ood_keys),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_leakage(splits: dict, records: list[dict]) -> dict:
    """
    Check that:
    1. No family appears in both train and test_ood
    2. Every compound key appears in exactly one split
    3. Frame counts are nonzero in all splits
    4. Frame balance is sensible (train >= val)
    """
    key_to_info = {r["compound_key"]: r for r in records}

    def families_of(keys):
        return set(
            key_to_info[k]["family"]
            for k in keys
            if k in key_to_info
        )

    train_fams   = families_of(splits["train_id"])
    val_fams     = families_of(splits["val_id"])
    test_id_fams = families_of(splits["test_id_same_family"])
    ood_fams     = families_of(splits["test_ood_family"])

    train_ood_overlap = train_fams & ood_fams

    all_keys = (
        splits["train_id"]
        + splits["val_id"]
        + splits["test_id_same_family"]
        + splits["test_ood_family"]
    )
    # Compound keys must be unique across splits
    from collections import Counter
    key_counts = Counter(all_keys)
    duplicates = [k for k, c in key_counts.items() if c > 1]

    fc = splits.get("frame_counts", {})
    nonzero_frames = all(v > 0 for v in fc.values()) if fc else False
    frame_balance_ok = fc.get("train", 0) >= fc.get("val", 0)

    report = {
        "leakage_free":         len(train_ood_overlap) == 0,
        "train_ood_overlap":    sorted(train_ood_overlap),
        "no_duplicate_rxns":    len(duplicates) == 0,
        "duplicate_compound_keys": list(set(duplicates)),
        "nonzero_frames":       nonzero_frames,
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
            and nonzero_frames
        ),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build leakage-safe family splits for Transition1x.")
    p.add_argument("--h5",              required=True, help="Path to Transition1x.h5")
    p.add_argument("--out",             default="splits.json")
    p.add_argument("--seed",            type=int, default=17)
    p.add_argument("--ood-fraction",    type=float, default=0.05)
    p.add_argument("--max-reactions",   type=int, default=None,
                   help="Cap reactions per split (pilot mode)")
    p.add_argument("--verify",          metavar="SPLITS_JSON",
                   help="Verify an existing splits.json")
    a = p.parse_args()

    print(f"[split] Discovering reactions in {a.h5}...")
    records = discover_reactions(a.h5)
    print(f"[split] Found {len(records)} reactions across "
          f"{len(set(r['family'] for r in records))} families.")

    if a.verify:
        with open(a.verify) as f:
            splits = json.load(f)
        report = verify_leakage(splits, records)
        print(json.dumps(report, indent=2))
        if report["PASS"]:
            print("[split] VERIFICATION PASSED")
        else:
            print("[split] VERIFICATION FAILED — see report above.")
        return

    total_frames = sum(r["n_frames"] for r in records)
    print(f"[split] Total frames in dataset: {total_frames}")

    splits = build_family_split(
        records,
        seed=a.seed,
        ood_frac=a.ood_fraction,
        train_frac=0.70,
        val_frac=0.15,
        test_id_frac=1.0 - 0.70 - 0.15 - a.ood_fraction,
        max_reactions_per_split=a.max_reactions,
    )

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

    fc = splits["frame_counts"]
    if not report["nonzero_frames"]:
        raise RuntimeError(
            f"[split] FATAL: one or more splits have zero frames: {fc}\n"
            f"Check --max-reactions or HDF5 structure."
        )

    if fc["train"] < fc["val"]:
        print(
            f"[split] WARNING: train ({fc['train']}) < val ({fc['val']}) frames. "
            f"Consider --max-reactions to balance."
        )


if __name__ == "__main__":
    main()
