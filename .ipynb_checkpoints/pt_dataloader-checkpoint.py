import os
import glob
import torch
from torch.utils.data import Dataset


class PtDataloader(Dataset):
    def __init__(self, txt_file, use_depth=True, color_transform=None):
        """
        txt_file: 每行一个 .pt 文件路径
        use_depth: 是否读取 depth 通道
        """
        with open(txt_file, "r") as f:
            self.files = [line.strip() for line in f if line.strip()]

        self.use_depth = use_depth
        self.color_transform = color_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        data = torch.load(path, map_location="cpu")

        if self.color_transform is not None:
            # 先统一成 float32，避免 transforms 对 half 出问题
            color = data["color"].to(torch.float32)
            color = self.color_transform(color)
        else:
            # 至少转成 float32（后面模型和 Normalize 都喜欢这个）
            color = data["color"].to(torch.float32)

        label = data["label"].to(torch.float32)   # [2]
        
        if self.use_depth:
            depth = data["depth"].to(torch.float32)   # float16/float32, [1,H,W]
            return color, depth, label
        else:
            return color, label
