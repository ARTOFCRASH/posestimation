import os
import glob
import torch
from torch.utils.data import Dataset


class PtDataloader(Dataset):
    def __init__(self, txt_file, use_depth=True):
        """
        txt_file: 每行一个 .pt 文件路径
        use_depth: 是否读取 depth 通道
        """
        with open(txt_file, "r") as f:
            self.files = [line.strip() for line in f if line.strip()]

        self.use_depth = use_depth

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        data = torch.load(path, map_location="cpu")

        color = data["color"].to(torch.float32)   # [3,H,W], [0,1]
        label = data["label"].to(torch.float32)   # [2]
        if self.use_depth:
            depth = data["depth"].to(torch.float32)   # float16/float32, [1,H,W]
            return color, depth, label
        else:
            return color, label
