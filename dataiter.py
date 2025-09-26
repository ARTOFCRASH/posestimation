from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np


class MyData(Dataset):
    def __init__(self, img_dir, trans=None, use_depth=True, roll=True, pitch=True):
        self.img_dir = img_dir
        self.transform = trans
        # 只收集 color 图，避免重复
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('_color.png')]
        self.use_depth = use_depth
        self.roll = roll
        self.pitch = pitch

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        rgb_img = Image.open(img_path).convert('RGB')

        # 对应的灰度图
        if self.use_depth:
            depth_name = img_name.replace('_color.png', '_depth.png')
            depth_path = os.path.join(self.img_dir, depth_name)
            depth_img = Image.open(depth_path).convert('L')  # 单通道灰度

        # 变换
        if self.transform:
            rgb_img = self.transform(rgb_img)      # [3,H,W]
            if self.use_depth:
                depth_img = self.transform(depth_img)  # [1,H,W]

        if self.use_depth:
            image = torch.cat([rgb_img, depth_img], dim=0)  # [4,H,W] 合并
        else:
            image = rgb_img

        # 从文件名解析标签
        # 命名: x_y_z_p16_color.png
        label_values = img_name.split('.')[0].split('_')[:2]
        roll, pitch = map(float, label_values)

        # 归一化到 [-1,1]（因为范围是 -50°~50°）
        roll = np.interp(roll, [-50, 50], [-1, 1])
        pitch = np.interp(pitch, [-50, 50], [-1, 1])

        labels = []
        if self.roll:
            labels.append(roll)
        if self.pitch:
            labels.append(pitch)

        return image, torch.tensor(labels, dtype=torch.float32)

