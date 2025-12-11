import os
import random
from pathlib import Path

import webdataset as wds

# ================== 配置区域 ==================
# 所有 npz 数据所在的根目录（下面可以有 p1_m_npz, p2_m_npz ...）
NPZ_ROOT = Path("/root/autodl-tmp/npz_dataset_cropped256")

# 输出 WebDataset 的目录
OUT_ROOT = Path("/root/autodl-tmp/wds_kaki")

# train/val 划分比例
TRAIN_RATIO = 0.8

# 每个 tar 里放多少个样本（可以根据数据量调整）
SHARD_SIZE = 5000

# 随机种子（保证可复现的划分）
RANDOM_SEED = 42
# =============================================


def collect_all_npz(root: Path):
    """递归收集 root 下的所有 .npz 文件路径"""
    npz_files = sorted(root.rglob("*.npz"))
    return npz_files


def split_train_val(files, train_ratio=0.8, seed=42):
    """打乱后按比例切分为 train / val"""
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:]

    return train_files, val_files


def write_sharded_webdataset(split_name, files, out_root, shard_size=5000):
    """
    把 files 列表打成若干个 tar：
    例如：
        train-000000.tar
        train-000001.tar
    """
    out_root.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    sample_idx = 0  # 全局递增，用来生成 __key__
    n_files = len(files)

    def open_new_shard(split_name, shard_idx):
        shard_path = out_root / f"{split_name}-{shard_idx:06d}.tar"
        print(f"[OPEN] {shard_path}")
        return wds.TarWriter(str(shard_path))

    if n_files == 0:
        print(f"[WARN] split '{split_name}' has 0 files, skip.")
        return

    sink = open_new_shard(split_name, shard_idx)
    num_in_shard = 0

    for path in files:
        path = Path(path)
        # __key__：样本 ID，保证在整个 split 内唯一
        key = f"{sample_idx:09d}"  # 000000001 这种格式

        with open(path, "rb") as f:
            npz_bytes = f.read()

        sample = {
            "__key__": key,
            "npz": npz_bytes,  # 会写成 000000001.npz
        }

        sink.write(sample)
        sample_idx += 1
        num_in_shard += 1

        # 当前 shard 满了，切换到下一个 shard
        if num_in_shard >= shard_size:
            sink.close()
            shard_idx += 1
            num_in_shard = 0
            sink = open_new_shard(split_name, shard_idx)

    # 最后一个 shard 可能没满，照样关掉
    if sink is not None:
        sink.close()

    print(f"[DONE] split='{split_name}', files={n_files}, shards={shard_idx + 1}")


def main():
    print(f"[INFO] NPZ_ROOT = {NPZ_ROOT}")
    all_npz = collect_all_npz(NPZ_ROOT)
    print(f"[INFO] Found total npz files: {len(all_npz)}")

    train_files, val_files = split_train_val(
        all_npz, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED
    )
    print(f"[INFO] Train: {len(train_files)}, Val: {len(val_files)}")

    # 输出目录结构：
    # OUT_ROOT/
    #   train-000000.tar
    #   train-000001.tar
    #   ...
    #   val-000000.tar
    #   ...
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    write_sharded_webdataset("train", train_files, OUT_ROOT, SHARD_SIZE)
    write_sharded_webdataset("val", val_files, OUT_ROOT, SHARD_SIZE)


if __name__ == "__main__":
    main()
