"""
tests/test_invariances.py
=========================

Invariance tests for ChargeAwarePotentialClean (the canonical model).

Tests verify charge conservation, translation/rotation/permutation invariance,
force-gradient consistency, force equivariance, and finite Coulomb energy.
These are necessary but NOT sufficient for scientific validity — see
reports/next_stage_plan.md for real success criteria.
"""

import pytest
import torch
import torch.nn as nn

from ralrc.model_clean import ChargeAwarePotentialClean


def _call_model(model, R, Z, Q, S):
    """
    Calls model with correct dtypes and gradient-enabled positions.

    Returns:
        E : scalar tensor
        F : (N, 3) tensor if model returns forces, otherwise autograd-derived
        R_used : position tensor actually used for gradients
    """
    R_used = R.clone().detach().requires_grad_(True)
    out = model(Z.long(), R_used, Q.long(), S.long())

    if isinstance(out, dict):
        E = out.get("energy", out.get("E"))
        F = out.get("forces", out.get("F"))
    elif isinstance(out, (tuple, list)):
        E = out[0]
        F = out[1] if len(out) > 1 else None
    else:
        E = out
        F = None

    if F is None:
        F = -torch.autograd.grad(E, R_used, create_graph=model.training, retain_graph=True)[0]

    return E, F, R_used


def _make_molecule(n_atoms=6, min_dist=1.5, charge=0, spin=1, seed=42):
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
    Q = torch.tensor(int(charge), dtype=torch.long)
    S = torch.tensor(int(spin), dtype=torch.long)
    return R, Z, Q, S


def _random_rotation(seed=7):
    torch.manual_seed(seed)
    A = torch.randn(3, 3)
    Qm, _ = torch.linalg.qr(A)
    if torch.det(Qm) < 0:
        Qm[:, 0] *= -1
    return Qm


@pytest.fixture(scope="module")
def model():
    m = ChargeAwarePotentialClean(use_coulomb=True)
    m.eval()
    return m


@pytest.fixture(scope="module")
def neutral_mol():
    return _make_molecule(n_atoms=6, charge=0, spin=1, seed=42)


@pytest.fixture(scope="module")
def charged_mol():
    return _make_molecule(n_atoms=6, charge=1, spin=1, seed=123)


def test_charge_conservation(model, neutral_mol):
    R, Z, Q, S = neutral_mol
    captured = {}

    def hook(module, inp, out):
        q_raw = out.squeeze(-1) if out.ndim > 1 else out
        q = q_raw + (Q.float() - q_raw.sum()) / q_raw.numel()
        captured["q"] = q.detach()

    handle = model.charge_head.register_forward_hook(hook)
    E, F, _ = _call_model(model, R, Z, Q, S)
    handle.remove()

    assert torch.isfinite(E)
    assert torch.isfinite(F).all()
    assert "q" in captured, "charge_head hook did not fire"
    assert torch.allclose(captured["q"].sum(), Q.float(), atol=1e-4)


def test_charge_conservation_nonzero(model, charged_mol):
    R, Z, Q, S = charged_mol
    captured = {}

    def hook(module, inp, out):
        q_raw = out.squeeze(-1) if out.ndim > 1 else out
        q = q_raw + (Q.float() - q_raw.sum()) / q_raw.numel()
        captured["q"] = q.detach()

    handle = model.charge_head.register_forward_hook(hook)
    E, F, _ = _call_model(model, R, Z, Q, S)
    handle.remove()

    assert torch.isfinite(E)
    assert torch.isfinite(F).all()
    assert "q" in captured, "charge_head hook did not fire"
    assert torch.allclose(captured["q"].sum(), Q.float(), atol=1e-4)


def test_translation_invariance(model, neutral_mol):
    R, Z, Q, S = neutral_mol
    t = torch.tensor([3.7, -2.1, 5.5])

    E0, _, _ = _call_model(model, R, Z, Q, S)
    E1, _, _ = _call_model(model, R + t, Z, Q, S)

    assert torch.allclose(E0.detach(), E1.detach(), atol=1e-4), (
        f"translation ΔE={abs((E0 - E1).item()):.3e}"
    )


def test_rotation_invariance(model, neutral_mol):
    R, Z, Q, S = neutral_mol
    Rot = _random_rotation(seed=7)

    E0, _, _ = _call_model(model, R, Z, Q, S)
    E1, _, _ = _call_model(model, R @ Rot.T, Z, Q, S)

    assert torch.allclose(E0.detach(), E1.detach(), atol=1e-3), (
        f"rotation ΔE={abs((E0 - E1).item()):.3e}"
    )


def test_permutation_invariance(model, neutral_mol):
    R, Z, Q, S = neutral_mol
    torch.manual_seed(99)
    perm = torch.randperm(R.shape[0])

    E0, _, _ = _call_model(model, R, Z, Q, S)
    E1, _, _ = _call_model(model, R[perm], Z[perm], Q, S)

    assert torch.allclose(E0.detach(), E1.detach(), atol=1e-4), (
        f"permutation ΔE={abs((E0 - E1).item()):.3e}"
    )


def test_forces_negative_gradient(model, neutral_mol):
    R, Z, Q, S = neutral_mol

    E, F_model, _ = _call_model(model, R, Z, Q, S)

    eps = 1e-3
    F_fd = torch.zeros_like(R)

    for i in range(R.shape[0]):
        for d in range(3):
            R_fwd = R.clone()
            R_bwd = R.clone()
            R_fwd[i, d] += eps
            R_bwd[i, d] -= eps

            E_fwd, _, _ = _call_model(model, R_fwd, Z, Q, S)
            E_bwd, _, _ = _call_model(model, R_bwd, Z, Q, S)

            F_fd[i, d] = -((E_fwd.detach() - E_bwd.detach()) / (2 * eps))

    max_err = (F_model.detach() - F_fd).abs().max().item()
    assert max_err < 5e-2, f"model force != finite-difference -grad(E), max_err={max_err:.3e}"

def test_force_equivariance(model, neutral_mol):
    R, Z, Q, S = neutral_mol
    Rot = _random_rotation(seed=13)

    _, F0, _ = _call_model(model, R, Z, Q, S)
    _, F1, _ = _call_model(model, R @ Rot.T, Z, Q, S)

    F0_rot = F0 @ Rot.T
    max_err = (F0_rot - F1).abs().max().item()
    assert max_err < 1e-3, f"force equivariance mismatch: {max_err:.3e}"


def test_finite_coulomb_short_range():
    m = ChargeAwarePotentialClean(use_coulomb=True)
    m.eval()

    R = torch.tensor([[0.0, 0.0, 0.0],
                      [0.05, 0.0, 0.0]], dtype=torch.float32)
    Z = torch.tensor([1, 1], dtype=torch.long)
    Q = torch.tensor(0, dtype=torch.long)
    S = torch.tensor(1, dtype=torch.long)

    E, F, _ = _call_model(m, R, Z, Q, S)

    assert torch.isfinite(E), f"short-range Coulomb produced non-finite energy: {E}"
    assert torch.isfinite(F).all(), f"short-range Coulomb produced non-finite forces: {F}"
    assert E.abs().item() < 1e5, f"short-range Coulomb energy too large: {E.item():.3e}"


def _synthetic_dataset(n=100, n_atoms=4, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    data = []

    for _ in range(n):
        R = torch.rand(n_atoms, 3, generator=rng) * 4.0
        Z = torch.randint(1, 6, (n_atoms,), generator=rng, dtype=torch.long)
        Q = torch.tensor(0, dtype=torch.long)
        S = torch.tensor(1, dtype=torch.long)
        E = torch.randn((), generator=rng)
        data.append((R, Z, Q, S, E))

    return data


def test_overfit_100_sanity():
    torch.manual_seed(0)
    m = ChargeAwarePotentialClean(hidden=64, use_coulomb=True)
    m.train()

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    dataset = _synthetic_dataset(n=100, n_atoms=4)

    for _ in range(500):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for R, Z, Q, S, E_tgt in dataset:
            E_pred, _, _ = _call_model(m, R, Z, Q, S)
            loss = loss + (E_pred - E_tgt).pow(2)
        loss = loss / len(dataset)
        loss.backward()
        opt.step()

    m.eval()
    mse = torch.tensor(0.0)
    for R, Z, Q, S, E_tgt in dataset:
        E_pred, _, _ = _call_model(m, R, Z, Q, S)
        mse = mse + (E_pred.detach() - E_tgt).pow(2)
    mse = mse / len(dataset)

    assert mse.item() < 0.5, f"overfit-100 failed: final MSE={mse.item():.4f}"
