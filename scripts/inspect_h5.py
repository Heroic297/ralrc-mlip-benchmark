import argparse
import h5py
from pathlib import Path

def describe_obj(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET {name}")
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")
        if obj.attrs:
            print("  attrs:")
            for k, v in obj.attrs.items():
                print(f"    {k}: {v}")
    elif isinstance(obj, h5py.Group):
        print(f"GROUP   {name}")
        if obj.attrs:
            print("  attrs:")
            for k, v in obj.attrs.items():
                print(f"    {k}: {v}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--max-items", type=int, default=300)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0

    def visitor(name, obj):
        nonlocal count
        if count >= args.max_items:
            return
        describe_obj(name, obj)
        count += 1

    with h5py.File(path, "r") as f:
        print(f"FILE: {path}")
        print("ROOT KEYS:", list(f.keys()))
        print()
        f.visititems(visitor)

if __name__ == "__main__":
    main()