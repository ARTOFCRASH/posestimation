import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import math


class MyDataset(Dataset):
    def __init__(self, files,
                 transform_color=None,
                 transform_depth=None,
                 use_depth=True):

        self.files = files
        if len(self.files) == 0:
            raise FileNotFoundError("Empty file list!")
        
        self.transform_color = transform_color
        self.transform_depth = transform_depth
        self.use_depth = use_depth


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=False)

        color = data["color"]
        depth = data["depth"]
        label = torch.from_numpy(data["label"]).float()
        
        color_t = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        
        if self.transform_color is not None:
                # 传进去的是 tensor，不是原来的 numpy
                color_t = self.transform_color(color_t)

        if self.use_depth:
            if self.transform_depth is not None:
                depth_t = self.transform_depth(depth_t)
        else:
            depth_t = None

        if self.use_depth:
            return color_t, depth_t, label
        else:
            return color_t, label


if __name__ == "__main__":

    def load_file_list(txt_path):
        with open(txt_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

            
    train_list = load_file_list("train_files.txt")
    val_list   = load_file_list("val_files.txt")

    train_dataset = MyDataset(train_list, use_depth=True)
    val_dataset   = MyDataset(val_list, use_depth=True)

    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset:   {len(val_dataset)}")

    # 取一个样本打印信息
    color, depth, label = train_dataset[0]

    print("\nSingle sample:")
    print(f"  color.shape = {color.shape}, dtype = {color.dtype}")
    print(f"  depth.shape = {depth.shape}, dtype = {depth.dtype}")
    print(f"  label       = {label}, dtype = {label.dtype}")

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )


    def concat_color_depth_single(color_t, depth_t):
        """
        color_t: [3,H,W] float tensor
        depth_t: [1,H,W] float tensor
        返回：PIL Image，左右拼接： [ COLOR | DEPTH ]
        """
        # color: [3,H,W] -> [H,W,3] uint8
        color_np = (color_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_color = Image.fromarray(color_np)  # RGB

        # depth: [1,H,W] -> 灰度 uint8
        depth_np = depth_t[0].cpu().numpy()
        if depth_np.max() > 0:
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        else:
            depth_norm = depth_np
        depth_img = (depth_norm * 255).astype(np.uint8)
        img_depth = Image.fromarray(depth_img)  # L

        # 灰度转 3 通道
        img_depth_rgb = img_depth.convert("RGB")

        # 左右拼接
        W_total = img_color.width + img_depth_rgb.width
        H_max   = max(img_color.height, img_depth_rgb.height)
        concat_img = Image.new("RGB", (W_total, H_max))
        concat_img.paste(img_color, (0, 0))
        concat_img.paste(img_depth_rgb, (img_color.width, 0))

        return concat_img


    def show_batch_color_depth(color_batch, depth_batch, n_cols=4, show=True):
        """
        color_batch: [B,3,H,W]
        depth_batch: [B,1,H,W]
        n_cols: 一行放多少个样本（每个样本已经是 COLOR|DEPTH）
        show: 是否直接 .show()

        返回: grid_img (PIL Image)
        """
        B = color_batch.size(0)
        # 先把每个样本的 COLOR|DEPTH 拼出来
        imgs = []
        for i in range(B):
            img = concat_color_depth_single(color_batch[i], depth_batch[i])
            imgs.append(img)

        if len(imgs) == 0:
            return None

        # 假设每个拼好的图尺寸相同
        w_single, h_single = imgs[0].size

        n_rows = math.ceil(B / n_cols)
        grid_w = w_single * n_cols
        grid_h = h_single * n_rows

        grid_img = Image.new("RGB", (grid_w, grid_h))

        for idx, img in enumerate(imgs):
            row = idx // n_cols
            col = idx % n_cols
            x = col * w_single
            y = row * h_single
            grid_img.paste(img, (x, y))

        if show:
            grid_img.show()

        return grid_img


    for color, depth, label in train_loader:
        print("Batch:")
        print("  color:", color.shape)
        print("  depth:", depth.shape)
        print("  label:", label.shape)
        grid_img = show_batch_color_depth(color, depth, n_cols=4, show=True)
        grid_img.save("batch2.png")
        break