"""Active learning loop with deep ensemble uncertainty."""
import torch

def select_high_uncertainty(predictions, k=100):
    """predictions: list of tensors (one per ensemble member). Returns top-k indices by std."""
    stack = torch.stack(predictions)
    std = stack.std(dim=0)
    return torch.argsort(std, descending=True)[:k].tolist()
