"""Dataset loaders for Transition1x, SPICE, NPZ, extxyz."""
import torch
from torch.utils.data import Dataset

class MLIPDataset(Dataset):
    def __init__(self, structures):
        self.structures = structures
    def __len__(self): return len(self.structures)
    def __getitem__(self, i): return self.structures[i]

def load_npz(path):
    import numpy as np
    d = np.load(path, allow_pickle=True)
    out = []
    for i in range(len(d["Z"])):
        out.append({
            "Z": torch.tensor(d["Z"][i]),
            "R": torch.tensor(d["R"][i], dtype=torch.float32),
            "E": torch.tensor(d["E"][i], dtype=torch.float32),
            "F": torch.tensor(d["F"][i], dtype=torch.float32),
            "Q": torch.tensor(int(d["Q"][i])),
            "S": torch.tensor(int(d["S"][i])),
            "reaction_id": str(d["rxn"][i]),
            "family": str(d["fam"][i]),
        })
    return out
