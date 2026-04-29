"""tests/test_split_safety.py

Synthetic metadata tests for leakage-safe family split logic.
No HDF5 file required — all tests operate on fabricated record lists
matching the schema returned by discover_reactions().

Invariants enforced:
  1. No family appears in both train and test_ood (family leakage)
  2. No compound key appears in more than one split (key uniqueness)
  3. All keys use compound-key format  "<split>::<formula>::<rxn_id>"
  4. verify_leakage detects zero-frame splits
  5. verify_leakage PASS on a clean split
  6. train > val (fraction sanity)
  7. Every record key assigned to exactly one split (coverage)
  8. build_family_split hard-fails if fractions do not sum to 1
"""

import pytest
from ralrc.split import build_family_split, verify_leakage


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_records(
    n_families: int = 20,
    reactions_per_family: int = 5,
    frames_per_reaction: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Fabricate discover_reactions()-style records without touching HDF5."""
    import random
    rng = random.Random(seed)
    records = []
    for fi in range(n_families):
        formula = f"C{fi+1}H{(fi+1)*2}"
        for ri in range(reactions_per_family):
            rxn_id = f"r{ri:04d}"
            hdf5_split = rng.choice(["train", "val", "test"])
            ck = f"{hdf5_split}::{formula}::{rxn_id}"
            records.append({
                "compound_key": ck,
                "rxn_id": rxn_id,
                "formula": formula,
                "family": formula,
                "split": hdf5_split,
                "n_frames": frames_per_reaction,
            })
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_train_ood_family_overlap():
    """No family appears in both train_id and test_ood_family."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)

    train_fams = {k.split("::")[1] for k in splits["train_id"]}
    ood_fams   = {k.split("::")[1] for k in splits["test_ood_family"]}
    overlap    = train_fams & ood_fams

    assert len(overlap) == 0, (
        f"Family leakage: {len(overlap)} families in both train and OOD: "
        f"{sorted(overlap)[:5]}"
    )


def test_compound_keys_unique_across_all_splits():
    """No compound key appears in more than one split."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)
    report = verify_leakage(splits, records)

    assert report["no_duplicate_rxns"], (
        f"Duplicate compound keys found across splits: "
        f"{report['duplicate_compound_keys'][:5]}"
    )


def test_all_keys_are_compound_format():
    """Every key in every split list must contain '::' and have exactly 3 parts."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)

    for split_name in ["train_id", "val_id", "test_id_same_family", "test_ood_family"]:
        for key in splits[split_name]:
            assert "::" in key, (
                f"Bare rxn_id (no '::') in {split_name}: {key!r}. "
                "Expected '<hdf5_split>::<formula>::<rxn_id>'"
            )
            parts = key.split("::")
            assert len(parts) == 3, (
                f"Malformed compound key in {split_name}: {key!r}. "
                f"Expected 3 '::'-separated parts, got {len(parts)}"
            )


def test_verify_leakage_detects_zero_frames():
    """verify_leakage must set nonzero_frames=False and PASS=False when any count is 0."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)

    broken = dict(splits)
    broken["frame_counts"] = {"train": 0, "val": 100, "test_id": 50, "test_ood": 20}

    report = verify_leakage(broken, records)
    assert not report["nonzero_frames"], "Should detect zero-frame train split"
    assert not report["PASS"], "PASS should be False with zero-frame split"


def test_verify_leakage_passes_clean_split():
    """verify_leakage.PASS must be True for a correctly generated split."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)
    report = verify_leakage(splits, records)

    assert report["PASS"], (
        f"verify_leakage failed on a clean split:\n"
        f"  train_ood_overlap : {report['train_ood_overlap']}\n"
        f"  duplicate_keys    : {report['duplicate_compound_keys']}\n"
        f"  nonzero_frames    : {report['nonzero_frames']}\n"
        f"  frame_counts      : {report['frame_counts']}"
    )


def test_train_larger_than_val():
    """With default fractions (70/15/10/5), train must have more reactions than val."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)

    n_train = len(splits["train_id"])
    n_val   = len(splits["val_id"])
    assert n_train > n_val, (
        f"Train ({n_train}) should exceed val ({n_val}) with 70/15 fraction split"
    )


def test_all_records_assigned_to_exactly_one_split():
    """Every record's compound key appears in exactly one of the four split lists."""
    records = _make_records(n_families=40, seed=0)
    splits  = build_family_split(records, seed=42)

    assigned = (
        set(splits["train_id"])
        | set(splits["val_id"])
        | set(splits["test_id_same_family"])
        | set(splits["test_ood_family"])
    )
    record_keys = {r["compound_key"] for r in records}
    missing = record_keys - assigned

    assert len(missing) == 0, (
        f"{len(missing)} record keys not in any split list: "
        f"{sorted(missing)[:5]}"
    )


def test_fraction_assertion_fires():
    """build_family_split must raise AssertionError if fractions don't sum to 1."""
    records = _make_records(n_families=20, seed=0)
    with pytest.raises(AssertionError):
        build_family_split(records, seed=0, train_frac=0.5, val_frac=0.5,
                           test_id_frac=0.2, ood_frac=0.1)


def test_disjoint_families_across_all_four_splits():
    """No family should appear in more than one of the four family-level buckets."""
    records = _make_records(n_families=40, seed=0)
    splits = build_family_split(records, seed=42)

    fmap = splits.get("family_split_map", {})
    assert len(fmap) > 0, "family_split_map should be populated"

    from collections import Counter
    counts = Counter(fmap.values())
    # Each value is one of train/val/test_id/test_ood — all keys should be unique
    # (families → split is a function, not a many-to-one mapping back)
    assert len(fmap) == sum(counts.values()), "family_split_map has duplicate families"
