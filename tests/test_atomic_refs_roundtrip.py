"""Roundtrip test: load → subtract atomic refs → add back → identical to load.

Uses a mocked HDF5 layout via h5py in a tmpdir so the test does not depend
on the full Transition1x dataset.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ralrc.transition1x import Transition1xDataset, ENERGY_KEY, FORCES_KEY


def _write_mock_h5(path: Path) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        train = f.create_group("train")
        for formula, z in [("CH4", [6, 1, 1, 1, 1]), ("H2O", [8, 1, 1])]:
            frml = train.create_group(formula)
            for rxn in ["rxn0", "rxn1"]:
                rxn_grp = frml.create_group(rxn)
                z_arr = np.array(z, dtype=np.int32)
                n_atoms = len(z)
                n_frames = 3
                rxn_grp.create_dataset("atomic_numbers", data=z_arr)
                rxn_grp.create_dataset(
                    "positions", data=rng.standard_normal((n_frames, n_atoms, 3))
                )
                # Plausible eV-scale energies (Transition1x convention).
                e_base = -1102.0 if formula == "CH4" else -2079.0
                rxn_grp.create_dataset(
                    ENERGY_KEY, data=e_base + 0.01 * rng.standard_normal(n_frames)
                )
                rxn_grp.create_dataset(
                    FORCES_KEY, data=rng.standard_normal((n_frames, n_atoms, 3))
                )


def test_atomic_refs_roundtrip(tmp_path: Path) -> None:
    h5_path = tmp_path / "mock.h5"
    _write_mock_h5(h5_path)

    # Per-element refs in eV (arbitrary plausible values).
    refs_eV = {1: -13.6, 6: -1029.5, 8: -2042.0}

    ds_raw = Transition1xDataset(h5_path)
    ds_sub = Transition1xDataset(h5_path, atom_refs=refs_eV)

    assert len(ds_raw) == len(ds_sub) > 0

    max_diff = 0.0
    for i in range(len(ds_raw)):
        s_raw = ds_raw[i]
        s_sub = ds_sub[i]
        recon = s_sub["energy"] + s_sub["atom_ref_correction_ev"]
        max_diff = max(max_diff, abs(recon - s_raw["energy"]))

    assert max_diff < 1e-6, f"roundtrip diff {max_diff} eV exceeds 1e-6 eV"


def test_subtraction_actually_changes_energy(tmp_path: Path) -> None:
    """Smoke check: with non-zero refs, the residual must differ from raw."""
    h5_path = tmp_path / "mock.h5"
    _write_mock_h5(h5_path)

    refs_eV = {1: -13.6, 6: -1029.5, 8: -2042.0}
    ds_raw = Transition1xDataset(h5_path)
    ds_sub = Transition1xDataset(h5_path, atom_refs=refs_eV)

    s_raw = ds_raw[0]
    s_sub = ds_sub[0]
    assert abs(s_raw["energy"] - s_sub["energy"]) > 1.0  # eV-scale shift
