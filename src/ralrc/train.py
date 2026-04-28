"""Training script with energy + force loss."""
import argparse, torch, yaml, json, os
from .model import ChargeAwarePotential

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    cfg = yaml.safe_load(open(a.config))
    model = ChargeAwarePotential(
        use_charge=cfg.get("use_charge", True),
        use_coulomb=cfg.get("use_coulomb", True),
    )
    out_dir = os.path.join("runs", cfg.get("name", "model"), f"seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Training {cfg.get('name','model')} seed={a.seed} -> {out_dir}")
    print("STUB: implement loop with real Transition1x/SPICE data")
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    json.dump({"status": "stub_no_data", "seed": a.seed}, open(os.path.join(out_dir, "log.json"), "w"))

if __name__ == "__main__": main()
