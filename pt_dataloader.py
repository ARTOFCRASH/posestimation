import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np


def rot_mat(point, vector, t):
    """
    生成一个绕任意轴旋转的 4x4 旋转矩阵。

    Params:
    - point: 旋转轴经过的点，4维numpy数组，例如 [a, b, c, 1]
    - vector: 旋转轴的方向向量，单位向量，4维numpy数组，例如 [u, v, w, 0]
    - t: 旋转角度，单位为弧度

    返回:
    - 4x4 transformation matrix
    """
    u, v, w, _ = vector
    a, b, c, _ = point
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    one_minus_cos_t = 1 - cos_t

    matrix = np.array([
        [
            u * u + (v * v + w * w) * cos_t,
            u * v * one_minus_cos_t - w * sin_t,
            u * w * one_minus_cos_t + v * sin_t,
            (a * (v * v + w * w) - u * (b * v + c * w)) * one_minus_cos_t + (b * w - c * v) * sin_t
        ],
        [
            u * v * one_minus_cos_t + w * sin_t,
            v * v + (u * u + w * w) * cos_t,
            v * w * one_minus_cos_t - u * sin_t,
            (b * (u * u + w * w) - v * (a * u + c * w)) * one_minus_cos_t + (c * u - a * w) * sin_t
        ],
        [
            u * w * one_minus_cos_t - v * sin_t,
            v * w * one_minus_cos_t + u * sin_t,
            w * w + (u * u + v * v) * cos_t,
            (c * (u * u + v * v) - w * (a * u + b * v)) * one_minus_cos_t + (a * v - b * u) * sin_t
        ],
        [0, 0, 0, 1]
    ])

    return matrix


def orientation_from_roll_pitch(roll, pitch):
    origin = np.array([0, 0, 0, 1], dtype=np.float32)
    x_axis = np.array([1, 0, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1, 0, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1, 0], dtype=np.float32)
    rollval = np.radians(roll)
    pitchval = np.radians(pitch)
    roll_mat = rot_mat(origin, y_axis, rollval)
    new_z_axis = roll_mat @ z_axis
    x_axis = roll_mat @ x_axis
    pitch_mat = rot_mat(origin, x_axis, pitchval)
    orientation = pitch_mat @ new_z_axis

    v = orientation[:3]
    v = v / np.linalg.norm(v)

    return v 


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
        
        label = label.view(-1)
        roll = float(label[0].item())
        pitch = float(label[1].item())
        v_np = orientation_from_roll_pitch(roll, pitch)
        v = torch.from_numpy(v_np).to(torch.float32)


        if self.use_depth:
            depth = data["depth"].to(torch.float32) # float16/float32, [1,H,W]
            if self.depth_transform is not None:
                depth = self.depth_transform(depth)
            return color, depth, v
        else:
            return color, v
