from pathlib import Path

p = Path("tests/test_invariances.py")
s = p.read_text(encoding="utf-8")

start = s.index("def test_forces_negative_gradient")
end = s.index("\ndef test_force_equivariance", start)

replacement = r'''def test_forces_negative_gradient(model, neutral_mol):
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
'''

s = s[:start] + replacement + s[end:]
p.write_text(s, encoding="utf-8")
print("patched test_forces_negative_gradient to use finite differences")
