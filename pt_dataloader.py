import os
import glob
import torch
from torch.utils.data import Dataset


class PtDataloader(Dataset):
    def __init__(self, src, use_depth=True, color_transform=None, depth_transform=None):
        """
        txt_file: 每行一个 .pt 文件路径
        use_depth: 是否读取 depth 通道
        """
        if src.endswith(".txt"):
            with open(src, "r") as f:
                self.files = [line.strip() for line in f if line.strip()]
        else:
            self.files = sorted(
                glob.glob(os.path.join(src, "**", "*.pt"), recursive=True)
            )
            
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {src}")

        self.use_depth = use_depth
        self.color_transform = color_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        data = torch.load(path, map_location="cpu")
        
        color = data["color"].to(torch.float32)
        if self.color_transform is not None:
            color = self.color_transform(color)

        label = data["label"].to(torch.float32)   # [2]
        
        if self.use_depth:
            depth = data["depth"].to(torch.float32) # float16/float32, [1,H,W]
            if self.depth_transform is not None:
                depth = self.depth_transform(depth)
            return color, depth, label
        else:
            return color, label
