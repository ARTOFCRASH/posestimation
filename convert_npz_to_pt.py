import os
import glob
import numpy as np
import torch
from tqdm import tqdm

SRC_ROOT = r"D:\zhou-yunong\files\persimmon data\RealSenseD405_raw\cropped256_npz"
DST_ROOT = r"D:\zhou-yunong\files\persimmon data\RealSenseD405_raw\cropped256_pt"
os.makedirs(DST_ROOT, exist_ok=True)


def main():
    npz_files = sorted(
        glob.glob(os.path.join(SRC_ROOT, "**", "*.npz"), recursive=True)
    )
    print(f"Found {len(npz_files)} npz files")

    for npz_path in tqdm(npz_files, desc="NPZ -> PT"):
        data = np.load(npz_path, allow_pickle=False)

        color_np = data["color"]   # (H, W, 3), uint8
        depth_np = data["depth"]   # (H, W)
        label_np = data["label"]   # (2,)

        color = (
            torch.from_numpy(color_np)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .half()
        )

        depth = torch.from_numpy(depth_np).unsqueeze(0).float().half()
        label = torch.from_numpy(label_np).float()

        # 保留相对路径结构（非常推荐）
        rel_path = os.path.relpath(npz_path, SRC_ROOT)
        pt_path = os.path.join(
            DST_ROOT,
            rel_path.replace(".npz", ".pt")
        )

        os.makedirs(os.path.dirname(pt_path), exist_ok=True)

        torch.save(
            {"color": color, "depth": depth, "label": label},
            pt_path
        )

    print("Done.")

if __name__ == "__main__":
    main()
