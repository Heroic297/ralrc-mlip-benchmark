"""ChargeAwarePotential: charge-conditioned MACE-style MLIP with shielded Coulomb."""
import torch
import torch.nn as nn

K_E = 14.3996  # eV * angstrom / e^2

class ChargeAwarePotential(nn.Module):
    def __init__(self, hidden=64, use_charge=True, use_coulomb=True, n_elements=119):
        super().__init__()
        self.use_charge = use_charge
        self.use_coulomb = use_coulomb
        self.embed = nn.Embedding(n_elements, hidden)
        self.q_embed = nn.Embedding(11, hidden)
        self.s_embed = nn.Embedding(11, hidden)
        self.energy_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.charge_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.shield = nn.Parameter(torch.zeros(n_elements, n_elements))

    def forward(self, Z, R, Q, S):
        R = R.requires_grad_(True)
        h = self.embed(Z) + self.q_embed(Q + 5).unsqueeze(0) + self.s_embed(S - 1).unsqueeze(0)
        rij = torch.cdist(R, R) + torch.eye(R.shape[0], device=R.device) * 1e6
        pair = torch.exp(-rij / 2.0)
        h = h + (pair @ h)
        E_local = self.energy_head(h).sum()
        E_total = E_local
        if self.use_charge:
            q_raw = self.charge_head(h).squeeze(-1)
            q = q_raw + (Q.float() - q_raw.sum()) / Z.shape[0]
            if self.use_coulomb:
                a = torch.nn.functional.softplus(self.shield[Z][:, Z])
                qq = q.unsqueeze(0) * q.unsqueeze(1)
                mask = torch.triu(torch.ones_like(rij), diagonal=1)
                E_coul = K_E * (qq * mask / torch.sqrt(rij**2 + a**2)).sum()
                E_total = E_total + E_coul
        F = -torch.autograd.grad(E_total, R, create_graph=self.training)[0]
        return E_total, F
