import numpy as np
import glob

root_dir = r"D:\files\persimmon data\npz dataset"

files = glob.glob(f"{root_dir}/**/*.npz", recursive=True)
files = files[:8000]  # 随便抽8000个

all_min = []
all_max = []

for f in files:
    data = np.load(f, allow_pickle=False)
    depth = data["depth"]  # (H,W)
    mask = depth > 0
    if mask.any():
        d_valid = depth[mask]
        all_min.append(d_valid.min())
        all_max.append(d_valid.max())

print("valid depth min over dataset:", np.min(all_min))
print("valid depth max over dataset:", np.max(all_max))
print("mean depth range:", np.mean(all_min), " ~ ", np.mean(all_max))
