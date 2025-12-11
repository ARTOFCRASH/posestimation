import os
import glob
import numpy as np
import torch
from tqdm import tqdm

SRC_ROOT = "/root/autodl-tmp/npz_dataset_cropped256"
DST_ROOT = "/root/autodl-tmp/pt_dataset"
os.makedirs(DST_ROOT, exist_ok=True)

MAX_DEPTH = 160.0  # 和你原来的 DepthNormalize 一致


def convert_one_folder(npz_dir, pt_dir):
    os.makedirs(pt_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    print(f"[{os.path.basename(npz_dir)}] {len(npz_files)} files")

    for npz_path in tqdm(npz_files, desc=os.path.basename(npz_dir)):
        data = np.load(npz_path, allow_pickle=False)

        color_np = data["color"]      # (H, W, 3), uint8
        depth_np = data["depth"]      # (H, W), uint16
        label_np = data["label"]      # (2,), float32

        # ---- 这里一步到位做完预处理 ----
        # color: HWC uint8 -> CHW float32 [0,1]
        color = torch.from_numpy(color_np).permute(2, 0, 1).float().div_(255.0).half()


        # depth: HW uint16 -> [1,H,W] float32 [0,1] (按 MAX_DEPTH 归一化)
        depth = torch.from_numpy(depth_np).unsqueeze(0).float().div_(MAX_DEPTH).clamp_(0,1).half()

        # label: float32 [2]
        label = torch.from_numpy(label_np).float()

        pt_path = os.path.join(
            pt_dir,
            os.path.basename(npz_path).replace(".npz", ".pt")
        )

        torch.save(
            {"color": color, "depth": depth, "label": label},
            pt_path,
        )


def main():
    subdirs = sorted(
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d))
    )

    print("Found subfolders:", subdirs)

    for sub in subdirs:
        src_dir = os.path.join(SRC_ROOT, sub)
        dst_dir = os.path.join(DST_ROOT, sub.replace("_npz", "_pt"))
        convert_one_folder(src_dir, dst_dir)


if __name__ == "__main__":
    main()
