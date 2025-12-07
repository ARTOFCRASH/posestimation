import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, files,
                 transform_color=None,
                 transform_depth=None,
                 use_depth=True,
                 target_size=256,
                 crop_and_resize=True):

        self.files = files
        if len(self.files) == 0:
            raise FileNotFoundError("Empty file list!")
        
        self.transform_color = transform_color
        self.transform_depth = transform_depth
        self.use_depth = use_depth
        self.target_size = target_size
        self.crop_and_resize = crop_and_resize


    def __len__(self):
        return len(self.files)


    def _crop_and_resize(self, color, depth):
        """
        color: np.ndarray, (H, W, 3), uint8
        depth: np.ndarray, (H, W),    uint16 / float
        返回: torch.Tensor color[3,target_size,target_size], depth[1,target_size,target_size]
        """
        H, W, _ = color.shape

        # 1) HWC -> CHW
        color_t = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()              # [1,H,W]

        # ------- 情况 1：不要 crop，只做整体 resize -------
        if not self.crop_and_resize:
            color_t = F.interpolate(
                color_t.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            depth_t = F.interpolate(
                depth_t.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="nearest"
            ).squeeze(0)

            return color_t, depth_t

        # ------- 情况 2：要 crop，用 depth > 0 找 mask -------
        mask = depth_t[0] > 0  # [H,W]，如果背景深度不是 0，这里改条件

        if not mask.any():
            # 没有任何前景像素 => 退路：直接整体 resize，不要递归自己
            color_t = F.interpolate(
                color_t.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            depth_t = F.interpolate(
                depth_t.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="nearest"
            ).squeeze(0)

            return color_t, depth_t

        ys, xs = torch.where(mask)
        y_min, y_max = ys.min().item(), ys.max().item()
        x_min, x_max = xs.min().item(), xs.max().item()

        h = y_max - y_min + 1
        w = x_max - x_min + 1

        side = int(max(h, w) * 1.2)  # margin = 20%
        if side < 1:
            side = max(h, w)

        cy = (y_min + y_max) / 2.0
        cx = (x_min + x_max) / 2.0

        y1 = int(round(cy - side / 2))
        y2 = y1 + side
        x1 = int(round(cx - side / 2))
        x2 = x1 + side

        # clip 到图像范围内
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(H, y2); x2 = min(W, x2)

        # ------- 3) crop 出矩形区域 -------
        color_crop = color_t[:, y1:y2, x1:x2]   # [3,h',w']
        depth_crop = depth_t[:, y1:y2, x1:x2]   # [1,h',w']

        h_crop = color_crop.shape[1]
        w_crop = color_crop.shape[2]

        # 极端保护：防止 h' 或 w' = 0
        if h_crop == 0 or w_crop == 0:
            color_crop = color_t
            depth_crop = depth_t
            h_crop = H
            w_crop = W

        # ------- 4) pad 成正方形 -------
        side2 = max(h_crop, w_crop)
        pad_h = side2 - h_crop
        pad_w = side2 - w_crop

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad 的顺序是 (left, right, top, bottom)
        color_pad = F.pad(
            color_crop,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0
        )  # [3, side2, side2]

        depth_pad = F.pad(
            depth_crop,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0
        )  # [1, side2, side2]

        # ------- 5) resize 到 target_size x target_size -------
        color_resized = F.interpolate(
            color_pad.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        depth_resized = F.interpolate(
            depth_pad.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="nearest"
        ).squeeze(0)

        return color_resized, depth_resized


    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=False)

        color = data["color"]
        depth = data["depth"]
        label = torch.from_numpy(data["label"]).float()

        color_t, depth_t = self._crop_and_resize(color, depth)

        if self.transform_color is not None:
                # 注意：这里传进去的是 tensor，不是原来的 numpy
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

    for color, depth, label in train_loader:
        print("Batch:")
        print("  color:", color.shape)
        print("  depth:", depth.shape)
        print("  label:", label.shape)
        break