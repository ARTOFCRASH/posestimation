# chunk_dataset.py
import os
import json
import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Chunked_Dataset(Dataset):
    def __init__(self,
                 chunks_dir,
                 transform_color=None,
                 transform_depth=None,
                 use_depth=True):
        """
        chunks_dir: 上一步生成的目录, 里面有 meta.json 和 chunk_*.pt
        """
        self.chunks_dir = Path(chunks_dir)
        meta_path = self.chunks_dir / "meta.json"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.chunk_infos = meta["chunks"]  # list of {file, num_samples}
        self.chunk_files = [c["file"] for c in self.chunk_infos]
        self.chunk_sizes = [c["num_samples"] for c in self.chunk_infos]

        # 前缀和，方便 idx → (chunk_id, offset) 查找
        self.cum_sizes = np.cumsum(self.chunk_sizes)
        self.total_size = int(self.cum_sizes[-1])

        self.transform_color = transform_color
        self.transform_depth = transform_depth
        self.use_depth = use_depth

        # 简单缓存：当前已加载的 chunk
        self._loaded_chunk_id = None
        self._loaded_chunk = None

    def __len__(self):
        return self.total_size

    def _get_chunk_index(self, idx):
        # idx: 0-based 全局索引
        chunk_id = int(bisect.bisect_right(self.cum_sizes, idx))
        # 例如 cum_sizes = [4096,8192,...]，idx=5000 → chunk_id=1
        prev_cum = 0 if chunk_id == 0 else self.cum_sizes[chunk_id - 1]
        offset = idx - prev_cum
        return chunk_id, int(offset)

    def _load_chunk(self, chunk_id):
        if self._loaded_chunk_id == chunk_id and self._loaded_chunk is not None:
            return

        chunk_file = self.chunk_files[chunk_id]
        chunk_path = self.chunks_dir / chunk_file

        data = torch.load(chunk_path, map_location="cpu")
        self._loaded_chunk = data
        self._loaded_chunk_id = chunk_id

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.total_size + idx
        if idx < 0 or idx >= self.total_size:
            raise IndexError

        chunk_id, offset = self._get_chunk_index(idx)
        self._load_chunk(chunk_id)

        data = self._loaded_chunk

        color = data["color"][offset]   # [3,H,W], dtype=uint8 or float
        depth = data["depth"][offset]   # [1,H,W]
        label = data["label"][offset]   # [2]

        # 保持和你原来 MyDataset 一样的约定：
        # color: float32, [0,1]
        # depth: float32 (后面再归一化)
        color = color.to(torch.float32) / 255.0

        depth = depth.to(torch.float32)

        if self.transform_color is not None:
            color = self.transform_color(color)

        if self.use_depth and self.transform_depth is not None:
            depth = self.transform_depth(depth)

        if self.use_depth:
            return color, depth, label
        else:
            return color, label
