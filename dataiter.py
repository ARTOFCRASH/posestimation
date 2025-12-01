import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, npz_dir,
                 transform_color =  None,
                 transform_depth = None,
                 use_depth = True):

        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No npz files in {npz_dir}")
        
        self.transform_color = transform_color
        self.transform_depth = transform_depth
        self.use_depth = use_depth


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=False)

        # ----- color -----
        color = data["color"]      # (H, W, 3), uint8
        if self.transform_color:
            color = self.transform_color(color)
        else:
            # HWC uint8 -> CHW float32
            color = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0

        # ----- depth -----
        if self.use_depth:
                    depth = data["depth"]   # (H, W)
                    if self.transform_depth:
                        depth = self.transform_depth(depth)
                    else:
                        depth = torch.from_numpy(depth).unsqueeze(0).float()
        else:
            depth = None

        # ----- label -----
        label = torch.from_numpy(data["label"]).float()  # [roll, pitch]

        if self.use_depth:
            return color, depth, label
        else:
            return color, label


if __name__ == "__main__":
    npz_dir = r"D:\files\npz dataset\p103_m_npz"

    
    dataset = MyDataset(npz_dir=npz_dir, use_depth=False)

    print(f"Total samples: {len(dataset)}")

    color, label = dataset[0]
    print("Single sample:")
    print(f"  color.shape = {color.shape}, dtype = {color.dtype}")
    # print(f"  depth.shape = {depth.shape}, dtype = {depth.dtype}")
    print(f"  label       = {label}, dtype = {label.dtype}")


    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )

    for batch_idx, (c, l) in enumerate(loader):
        print("\nFirst batch:")
        print(f"  color batch shape = {c.shape}")   # [B, 3, H, W]
        #print(f"  depth batch shape = {d.shape}")   # [B, 1, H, W]
        print(f"  label batch shape = {l.shape}")   # [B, 2]