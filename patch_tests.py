from pathlib import Path

p = Path("tests/test_invariances.py")
s = p.read_text(encoding="utf-8")

# Make molecule fixtures return spin multiplicity too.
s = s.replace("return pos, Z, torch.tensor(charge)", "return pos, Z, torch.tensor(charge), torch.tensor(1)")

# Update tuple unpacking.
s = s.replace("pos, Z, Q = neutral_mol", "pos, Z, Q, S = neutral_mol")
s = s.replace("pos, Z, Q = charged_mol", "pos, Z, Q, S = charged_mol")

# Update all model(pos, Z, Q) calls to model(Z, pos, Q, S).
s = s.replace("model(pos, Z, Q)", "model(Z, pos, Q, S)")
s = s.replace("model(pos + t, Z, Q)", "model(Z, pos + t, Q, S)")
s = s.replace("model(pos_rot, Z, Q)", "model(Z, pos_rot, Q, S)")
s = s.replace("model(pos[perm], Z[perm], Q)", "model(Z[perm], pos[perm], Q, S)")
s = s.replace("model(pos_ad, Z, Q)", "model(Z, pos_ad, Q, S)")
s = s.replace("model(p_fwd, Z, Q)", "model(Z, p_fwd, Q, S)")
s = s.replace("model(p_bwd, Z, Q)", "model(Z, p_bwd, Q, S)")
s = s.replace("model(pos_r_ad, Z, Q)", "model(Z, pos_r_ad, Q, S)")

# Fix finite-coulomb local S variable.
s = s.replace("Q   = torch.tensor(0.0)\n\n    with torch.no_grad():\n        E = m(Z, pos, Q, S)", "Q   = torch.tensor(0.0)\n    S   = torch.tensor(1)\n\n    with torch.no_grad():\n        E = m(Z, pos, Q, S)")

# Fix synthetic dataset to include S.
s = s.replace("data.append((pos, Z, Q, E))", "data.append((pos, Z, Q, torch.tensor(1), E))")
s = s.replace("for pos, Z, Q, E_tgt in dataset:", "for pos, Z, Q, S, E_tgt in dataset:")
s = s.replace("m(pos, Z, Q)", "m(Z, pos, Q, S)")

p.write_text(s, encoding="utf-8")
print("patched tests/test_invariances.py for model(Z, R, Q, S)")
