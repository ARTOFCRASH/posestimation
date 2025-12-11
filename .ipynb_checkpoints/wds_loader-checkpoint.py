import io
import numpy as np
import torch
import webdataset as wds


def decode_npz_sample(sample):
    # 这里假设 decode() 已经把 npz 解成 dict-like
    data = sample["npz"]
    # 有两种可能：np.lib.npyio.NpzFile 或者 dict，看你写入时的形式
    # 如果是 NpzFile，data["color"] / data["depth"] / data["label"] 也一样能访问

    color = data["color"]
    depth = data["depth"]
    label = data["label"]

    color = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0
    depth = torch.from_numpy(depth).unsqueeze(0).float()
    label = torch.from_numpy(label).float()

    sample["color"] = color
    sample["depth"] = depth
    sample["label"] = label
    return sample



def get_wds_loader(
    shards_pattern,
    batch_size=128,
    num_workers=8,
    shuffle=True,
    buffer_size=1000,
    use_depth=True,
):
    """
    shards_pattern 例如:
        '/root/autodl-tmp/wds_kaki/train-{000000..000168}.tar'
    """
    dataset = wds.WebDataset(shards_pattern, shardshuffle=False)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = (
        dataset
        .decode()                # 保持 decode 版本（此时 sample["npz"] 已是 np.load 结果）
        .map(decode_npz_sample)  # 写入 color / depth / label 三个键
    )

    if use_depth:
        # 模型用深度 → 返回 (color, depth, label)
        dataset = dataset.to_tuple("color", "depth", "label")
    else:
        # 模型不用深度 → 只返回 (color, label)
        dataset = dataset.to_tuple("color", "label")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return loader


if __name__ == "__main__":
    train_shards = "/root/autodl-tmp/wds_kaki/train-{000000..000168}.tar"
    val_shards   = "/root/autodl-tmp/wds_kaki/val-{000000..000042}.tar"
    use_depth = False
    train_loader = get_wds_loader(train_shards, batch_size=64, num_workers=4, shuffle=True, use_depth=use_depth)
    val_loader   = get_wds_loader(val_shards,   batch_size=64, num_workers=2, shuffle=False, use_depth=use_depth)
    if use_depth:
        for color, depth, label in train_loader:
            print("color:", color.shape, color.dtype)  # [B, 3, H, W]
            print("depth:", depth.shape, depth.dtype)  # [B, 1, H, W]
            print("label:", label.shape, label.dtype)  # [B, 2]
            break
    else:
        for color, label in train_loader:
            print("color:", color.shape, color.dtype)  # [B, 3, H, W]
            print("label:", label.shape, label.dtype)  # [B, 2]
            break
