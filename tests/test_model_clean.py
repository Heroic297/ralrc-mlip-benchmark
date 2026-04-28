"""
tests/test_model_clean.py
=========================
Clean tests for ChargeAwarePotentialClean.

Rules:
- Never wrap force-computing calls in torch.no_grad().
- Always pass Q and S as torch.long.
- Use finite differences to verify force consistency (avoids double-backward).
- Tests check symmetry algebra and output schema only.
- Passing these tests does NOT validate the scientific hypothesis.
  See reports/next_stage_plan.md for real success criteria.
"""

import pytest
import torch
from ralrc.model_clean import ChargeAwarePotentialClean

# ------------------------------------------------------------------ helpers

def _molecule(n_atoms=6, charge=0, spin=1, seed=42, min_dist=1.5):
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        R = torch.rand(n_atoms, 3, generator=rng) * 5.0
        diff = R.unsqueeze(1) - R.unsqueeze(0)
        dist = diff.pow(2).sum(-1).sqrt()
        dist.fill_diagonal_(999.0)
        if dist.min().item() >= min_dist:
            break
    Z = torch.randint(1, 9, (n_atoms,), generator=rng, dtype=torch.long)
    Q = torch.tensor(charge, dtype=torch.long)
    S = torch.tensor(spin,   dtype=torch.long)
    return R, Z, Q, S


def _rot(seed=7):
    torch.manual_seed(seed)
    A = torch.randn(3, 3)
    Qm, _ = torch.linalg.qr(A)
    if torch.det(Qm) < 0:
        Qm[:, 0] *= -1
    return Qm


def _call(model, R, Z, Q, S, compute_forces=True):
    return model(Z, R, Q, S, compute_forces=compute_forces)


def _fd_forces(model, R, Z, Q, S, eps=1e-3):
    F_fd = torch.zeros_like(R)
    for i in range(R.shape[0]):
        for d in range(3):
            Rfwd = R.clone(); Rfwd[i, d] += eps
            Rbwd = R.clone(); Rbwd[i, d] -= eps
            Efwd = _call(model, Rfwd, Z, Q, S, compute_forces=False)["energy"].detach()
            Ebwd = _call(model, Rbwd, Z, Q, S, compute_forces=False)["energy"].detach()
            F_fd[i, d] = -((Efwd - Ebwd) / (2 * eps))
    return F_fd


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def model():
    m = ChargeAwarePotentialClean(use_coulomb=True)
    m.eval()
    return m


@pytest.fixture(scope="module")
def mol():
    return _molecule(n_atoms=6, charge=0, spin=1, seed=42)


@pytest.fixture(scope="module")
def mol_charged():
    return _molecule(n_atoms=6, charge=1, spin=1, seed=123)


# ------------------------------------------------------------------ tests

def test_smoke_import():
    """model_clean.py is importable and instantiates without error."""
    m = ChargeAwarePotentialClean()
    assert m is not None


def test_output_schema(model, mol):
    """forward() returns a dict with required keys."""
    R, Z, Q, S = mol
    out = _call(model, R, Z, Q, S, compute_forces=True)
    assert isinstance(out, dict), "forward() must return a dict"
    assert "energy"  in out, "missing key: energy"
    assert "forces"  in out, "missing key: forces"
    assert "charges" in out, "missing key: charges"
    assert out["energy"].shape == torch.Size([])
    assert out["forces"].shape == R.shape
    assert out["charges"].shape == (R.shape[0],)


def test_output_schema_no_forces(model, mol):
    """compute_forces=False gives energy and charges but no forces key."""
    R, Z, Q, S = mol
    with torch.no_grad():
        out = _call(model, R, Z, Q, S, compute_forces=False)
    assert "energy"  in out
    assert "charges" in out
    assert "forces"  not in out


def test_charge_conservation_neutral(model, mol):
    """sum(q_i) == Q for neutral molecule."""
    R, Z, Q, S = mol
    q = model.predict_charges(Z, R, Q, S)
    assert torch.allclose(q.sum(), Q.float(), atol=1e-5), \
        f"charge not conserved: sum(q)={q.sum().item():.6f}, Q={Q.item()}"


def test_charge_conservation_charged(model, mol_charged):
    """sum(q_i) == Q for charged molecule."""
    R, Z, Q, S = mol_charged
    q = model.predict_charges(Z, R, Q, S)
    assert torch.allclose(q.sum(), Q.float(), atol=1e-5), \
        f"charge not conserved: sum(q)={q.sum().item():.6f}, Q={Q.item()}"


def test_translation_invariance(model, mol):
    """Energy unchanged under rigid translation."""
    R, Z, Q, S = mol
    t = torch.tensor([3.7, -2.1, 5.5])
    E0 = _call(model, R,     Z, Q, S, compute_forces=False)["energy"].detach()
    E1 = _call(model, R + t, Z, Q, S, compute_forces=False)["energy"].detach()
    assert torch.allclose(E0, E1, atol=1e-4), \
        f"translation Î”E={abs((E0 - E1).item()):.3e}"


def test_rotation_invariance(model, mol):
    """Energy unchanged under rigid rotation."""
    R, Z, Q, S = mol
    Rot = _rot(seed=7)
    E0 = _call(model, R,          Z, Q, S, compute_forces=False)["energy"].detach()
    E1 = _call(model, R @ Rot.T,  Z, Q, S, compute_forces=False)["energy"].detach()
    assert torch.allclose(E0, E1, atol=1e-3), \
        f"rotation Î”E={abs((E0 - E1).item()):.3e}"


def test_permutation_invariance(model, mol):
    """Energy unchanged under atom permutation."""
    R, Z, Q, S = mol
    torch.manual_seed(99)
    perm = torch.randperm(R.shape[0])
    E0 = _call(model, R,       Z,       Q, S, compute_forces=False)["energy"].detach()
    E1 = _call(model, R[perm], Z[perm], Q, S, compute_forces=False)["energy"].detach()
    assert torch.allclose(E0, E1, atol=1e-4), \
        f"permutation Î”E={abs((E0 - E1).item()):.3e}"


def test_forces_fd_consistency(model, mol):
    """Forces from forward() agree with finite-difference -dE/dR."""
    R, Z, Q, S = mol
    out = _call(model, R, Z, Q, S, compute_forces=True)
    F_model = out["forces"].detach()
    F_fd    = _fd_forces(model, R, Z, Q, S)
    max_err = (F_model - F_fd).abs().max().item()
    assert max_err < 5e-2, f"force FD mismatch: max_err={max_err:.3e}"


def test_force_equivariance(model, mol):
    """Forces transform as vectors under rotation."""
    R, Z, Q, S = mol
    Rot = _rot(seed=13)
    F0 = _call(model, R,         Z, Q, S, compute_forces=True)["forces"].detach()
    F1 = _call(model, R @ Rot.T, Z, Q, S, compute_forces=True)["forces"].detach()
    F0_rot = F0 @ Rot.T
    max_err = (F0_rot - F1).abs().max().item()
    assert max_err < 1e-3, f"force equivariance mismatch: {max_err:.3e}"


def test_finite_coulomb_short_range():
    """Shielded Coulomb must stay finite when atoms are very close."""
    m = ChargeAwarePotentialClean(use_coulomb=True)
    m.eval()
    R = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    Z = torch.tensor([1, 1], dtype=torch.long)
    Q = torch.tensor(0, dtype=torch.long)
    S = torch.tensor(1, dtype=torch.long)
    out = _call(m, R, Z, Q, S, compute_forces=False)
    E = out["energy"]
    assert torch.isfinite(E),        f"short-range E not finite: {E}"
    assert E.abs().item() < 1e5,     f"short-range E too large: {E.item():.3e}"


def test_dtype_guard():
    """Passing float Q or S must raise TypeError, not silently fail."""
    m = ChargeAwarePotentialClean()
    R = torch.rand(4, 3)
    Z = torch.randint(1, 9, (4,), dtype=torch.long)
    Q_float = torch.tensor(0.0, dtype=torch.float32)
    S = torch.tensor(1, dtype=torch.long)
    with pytest.raises(TypeError):
        m(Z, R, Q_float, S)


def test_overfit_small():
    """
    Model can memorize 20 random (R, E) pairs in 300 steps.
    Tests that the loss landscape is not degenerate.
    Final MSE must be < 0.2.
    """
    torch.manual_seed(0)
    m = ChargeAwarePotentialClean(hidden=64, use_coulomb=True)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    dataset = []
    rng = torch.Generator(); rng.manual_seed(1)
    for _ in range(20):
        R = torch.rand(4, 3, generator=rng) * 4.0
        Z = torch.randint(1, 6, (4,), generator=rng, dtype=torch.long)
        Q = torch.tensor(0, dtype=torch.long)
        S = torch.tensor(1, dtype=torch.long)
        E_tgt = torch.randn((), generator=rng)
        dataset.append((R, Z, Q, S, E_tgt))

    for _ in range(300):
        opt.zero_grad()
        loss = sum(
            (m(Z, R, Q, S, compute_forces=False)["energy"] - E_tgt).pow(2)
            for R, Z, Q, S, E_tgt in dataset
        ) / len(dataset)
        loss.backward()
        opt.step()

    m.eval()
    mse = sum(
        (m(Z, R, Q, S, compute_forces=False)["energy"].detach() - E_tgt).pow(2)
        for R, Z, Q, S, E_tgt in dataset
    ) / len(dataset)
    assert mse.item() < 0.2, f"overfit-20 failed: MSE={mse.item():.4f}"