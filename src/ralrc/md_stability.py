"""MD stability test: NVE/NVT, energy drift, bond explosions."""
import argparse, json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=50000)
    a = p.parse_args()
    print(f"MD stability test {a.checkpoint} for {a.steps} steps")
    print("STUB: 100ps NVE + NVT, velocity-Verlet, log E_drift")
    json.dump({"status": "stub_no_data", "steps_completed": 0,
               "exploded": None, "energy_drift": None},
              open("md_log.json", "w"), indent=2)

if __name__ == "__main__": main()
