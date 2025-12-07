import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
        if self.transform_color:
            color = self.transform_color(color)
        else:
            color = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0

        if self.use_depth:
            depth = data["depth"]
            if self.transform_depth:
                depth = self.transform_depth(depth)
            else:
                depth = torch.from_numpy(depth).unsqueeze(0).float()
        else:
            depth = None

        label = torch.from_numpy(data["label"]).float()  # [roll, pitch]

        if self.use_depth:
            return color, depth, label
        else:
            return color, label


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