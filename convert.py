from pathlib import Path

import numpy as np


root = Path("/Users/sophiakang/Documents/GitHub/MDF_AL")

npz_count = 0
for npz_path in root.rglob("*.npz"):
    with np.load(npz_path, allow_pickle=False) as _:
        pass
    npz_count += 1
    print(f"Validated: {npz_path}")

print(f"Done. Validated {npz_count} NPZ files.")
