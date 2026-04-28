"""Evaluation: energy/force MAE, barrier MAE, TS-force MAE, OOD degradation."""
import argparse, json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="metrics.json")
    a = p.parse_args()
    print(f"Eval {a.checkpoint}")
    print("STUB: compute energy_mae, force_mae, barrier_mae, ts_force_mae, ood_deg")
    metrics = {"status": "not_run_no_data", "energy_mae": None,
               "force_mae": None, "barrier_mae": None,
               "ts_force_mae": None, "ood_degradation": None}
    json.dump(metrics, open(a.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
