# convert_npz_to_chunks.py
import os
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_file_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def make_chunks(npz_list, out_dir, chunk_size=4096):
    """
    npz_list: 所有 npz 文件路径（list[str]）
    out_dir:  输出目录，会创建 chunk_00000.pt 等文件
    chunk_size: 每个 chunk 含多少样本
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"chunks": []}
    num_files = len(npz_list)
    num_chunks = math.ceil(num_files / chunk_size)

    print(f"Total samples: {num_files}, chunk_size={chunk_size}, "
          f"will create {num_chunks} chunks under {out_dir}")

    idx = 0
    chunk_id = 0

    while idx < num_files:
        batch_paths = npz_list[idx: idx + chunk_size]
        this_chunk_size = len(batch_paths)

        colors = []
        depths = []
        labels = []

        print(f"Building chunk {chunk_id} with {this_chunk_size} samples...")
        for p in tqdm(batch_paths):
            data = np.load(p)

            color = data["color"]    # [H,W,3], uint8
            depth = data["depth"]    # [H,W] 或 [H,W,1]
            label = data["label"]    # [2]

            # 统一成张量
            color = torch.from_numpy(color)              # H,W,3
            color = color.permute(2, 0, 1).contiguous()  # 3,H,W

            depth = torch.from_numpy(depth)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)               # 1,H,W
            elif depth.ndim == 3 and depth.shape[0] != 1:
                # 万一是 H,W,1 这种
                depth = depth.permute(2, 0, 1)          # 1,H,W

            label = torch.from_numpy(label).float()      # 2

            colors.append(color)
            depths.append(depth)
            labels.append(label)

        colors = torch.stack(colors, dim=0)   # [N,3,H,W]
        depths = torch.stack(depths, dim=0)   # [N,1,H,W]
        labels = torch.stack(labels, dim=0)   # [N,2]

        chunk_name = f"chunk_{chunk_id:05d}.pt"
        chunk_path = out_dir / chunk_name

        torch.save({
            "color": colors,   # 仍然是 uint8 (如果原来是)
            "depth": depths,   # 原 dtype
            "label": labels,   # float32
        }, chunk_path)

        meta["chunks"].append({
            "file": chunk_name,
            "num_samples": this_chunk_size
        })

        idx += this_chunk_size
        chunk_id += 1

    # 写 meta.json
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta to {meta_path}")


if __name__ == "__main__":
    # 根据你的实际路径改
    train_txt = "/root/autodl-tmp/project/pose_estimation/train_files.txt"
    val_txt   = "/root/autodl-tmp/project/pose_estimation/val_files.txt"

    train_list = load_file_list(train_txt)
    val_list   = load_file_list(val_txt)

    np.random.shuffle(train_list)
    np.random.shuffle(val_list)

    # 输出目录随便定一个
    make_chunks(train_list, out_dir="/root/autodl-tmp/chunks/train", chunk_size=4096)
    make_chunks(val_list,   out_dir="/root/autodl-tmp/chunks/val",   chunk_size=4096)
