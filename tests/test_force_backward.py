"""Regression test: train-mode forward + force loss must backward without error.

Bug: forward() called autograd.grad(E, R_diff, create_graph=True, retain_graph=False).
     loss.backward() then failed with "Trying to backward through the graph a second time"
     because E's graph was freed before the backward pass could use it.

Fix: retain_graph must equal create_graph.
"""
import pytest
import torch
import torch.nn as nn
from ralrc.model_clean import ChargeAwarePotentialClean


def _tiny_molecule(device):
    """Return (Z, R, Q, S) for a 4-atom H2O + H system."""
    Z = torch.tensor([8, 1, 1, 1], dtype=torch.long, device=device)
    R = torch.tensor([
        [0.0,  0.0,  0.0],
        [0.96, 0.0,  0.0],
        [-0.24, 0.93, 0.0],
        [1.5,  0.5,  0.0],
    ], dtype=torch.float32, device=device)
    Q = torch.zeros((), dtype=torch.long, device=device)
    S = torch.ones((), dtype=torch.long, device=device)
    return Z, R, Q, S


@pytest.mark.parametrize("use_coulomb", [False, True])
def test_train_force_backward_no_error(use_coulomb):
    """One train-mode forward + force loss backward must not raise."""
    device = torch.device("cpu")
    model = ChargeAwarePotentialClean(hidden=16, use_coulomb=use_coulomb)
    model.train()

    Z, R, Q, S = _tiny_molecule(device)
    R.requires_grad_(True)

    out = model.forward(Z, R, Q, S, compute_forces=True)

    assert "energy" in out
    assert "forces" in out, "forces must be present when compute_forces=True"
    assert out["forces"].shape == R.shape

    F_ref = torch.zeros_like(out["forces"])
    E_ref = torch.zeros((), device=device)

    loss = (out["energy"] - E_ref).abs() + 100.0 * (out["forces"] - F_ref).abs().mean()

    # This must NOT raise: "Trying to backward through the graph a second time"
    loss.backward()

    # At least one model parameter must have a non-None gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No model parameter received a gradient"


def test_eval_force_inference_no_leak():
    """Eval-mode force inference must work under no_grad without memory leak."""
    device = torch.device("cpu")
    model = ChargeAwarePotentialClean(hidden=16, use_coulomb=True)
    model.eval()

    Z, R, Q, S = _tiny_molecule(device)

    with torch.no_grad():
        with torch.enable_grad():
            R_g = R.detach().requires_grad_(True)
            out = model.forward(Z, R_g, Q, S, compute_forces=True)

    assert out["forces"].shape == R.shape
    # In eval no_grad context, forces tensor should have no grad_fn
    assert out["forces"].grad_fn is None or not out["forces"].requires_grad
