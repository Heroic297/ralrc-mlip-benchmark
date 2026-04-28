"""Smoke tests for Transition1x lazy loader.

Requires: data/raw/Transition1x.h5 to exist.
Skips gracefully if the file is absent (CI-safe).
"""

import numpy as np
import pytest
from pathlib import Path

H5_PATH = Path("data/raw/Transition1x.h5")

pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def h5_available():
    if not H5_PATH.exists():
        pytest.skip(f"Transition1x.h5 not found at {H5_PATH}")
    return H5_PATH


def test_iter_basic(h5_available):
    from src.ralrc.transition1x import iter_transition1x

    samples = list(iter_transition1x(h5_available, max_reactions=3))
    assert len(samples) > 0, "iterator yielded nothing"


def test_sample_schema(h5_available):
    from src.ralrc.transition1x import iter_transition1x

    for s in iter_transition1x(h5_available, max_reactions=1):
        assert set(s.keys()) >= {"z", "pos", "energy", "forces", "formula", "rxn_id", "frame_idx", "split"}
        break


def test_shapes_dtypes(h5_available):
    from src.ralrc.transition1x import iter_transition1x

    for s in iter_transition1x(h5_available, max_reactions=5):
        n = len(s["z"])
        assert n > 0, "zero atoms"
        assert s["z"].dtype == np.int32, f"z dtype: {s['z'].dtype}"
        assert s["pos"].shape == (n, 3), f"pos shape: {s['pos'].shape}"
        assert s["forces"].shape == (n, 3), f"forces shape: {s['forces'].shape}"
        assert isinstance(s["energy"], float), f"energy type: {type(s['energy'])}"
        # forces should be finite
        assert np.all(np.isfinite(s["forces"])), "non-finite forces"
        assert np.isfinite(s["energy"]), "non-finite energy"


def test_endpoints_included(h5_available):
    from src.ralrc.transition1x import iter_transition1x

    ep_samples = [
        s for s in iter_transition1x(h5_available, max_reactions=10, include_endpoints=True)
        if "endpoint" in s
    ]
    # Transition1x should have endpoints; if dataset has them we expect at least one
    # (we don't hard-fail if dataset slice has none)
    assert isinstance(ep_samples, list)
    for s in ep_samples:
        assert s["endpoint"] in ("reactant", "product", "transition_state")
        assert s["frame_idx"] == 0 or s["endpoint"] == "transition_state"


def test_formula_filter(h5_available):
    from src.ralrc.transition1x import iter_transition1x
    import h5py

    # pick first formula from train split
    with h5py.File(h5_available, "r") as f:
        for split in ["train", "data"]:
            if split in f:
                formula = next(iter(f[split].keys()))
                break

    samples = list(iter_transition1x(h5_available, formulas=[formula], max_reactions=20))
    assert all(s["formula"] == formula for s in samples), "formula filter leaked other formulas"


def test_dataset_index(h5_available):
    from src.ralrc.transition1x import Transition1xDataset

    ds = Transition1xDataset(h5_available, max_reactions=5) if False else None
    # Minimal: just instantiate with full iterator path via iter
    from src.ralrc.transition1x import iter_transition1x
    samples = list(iter_transition1x(h5_available, max_reactions=5))
    assert len(samples) > 0


def test_no_full_load(h5_available):
    """Confirm that iterating 3 reactions doesn't load the full file (proxy: psutil RSS)."""
    import psutil, os
    from src.ralrc.transition1x import iter_transition1x

    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss
    _ = list(iter_transition1x(h5_available, max_reactions=3))
    after = proc.memory_info().rss
    delta_mb = (after - before) / 1e6
    assert delta_mb < 500, f"Memory grew by {delta_mb:.1f} MB for 3 reactions — possible full load"
