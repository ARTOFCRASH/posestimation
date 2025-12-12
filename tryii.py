import time
import torch
from torch.utils.data import DataLoader

from pt_dataloader import PtDataloader  # 用你现在的 Dataset 类


# ====== 配置区域：按你当前训练的参数来 ======
TRAIN_TXT   = "pt_train_files.txt"  # 训练集文件列表
USE_DEPTH   = True                  # 和 train.py 里的 USE_DEPTH 保持一致
BATCH_SIZE  = 256                   # 和训练时一样
NUM_WORKERS = 8                     # 和训练时一样
PIN_MEMORY  = True                  # 和训练时一样

# 想测试多少个 batch 的速度（比如 200 个）
N_TEST_BATCHES = 200
# ==========================================


def main():
    print("=== Benchmark DataLoader I/O speed ===")
    print(f"TXT file     : {TRAIN_TXT}")
    print(f"USE_DEPTH    : {USE_DEPTH}")
    print(f"BATCH_SIZE   : {BATCH_SIZE}")
    print(f"NUM_WORKERS  : {NUM_WORKERS}")
    print(f"PIN_MEMORY   : {PIN_MEMORY}")
    print(f"TEST BATCHES : {N_TEST_BATCHES}")
    print("======================================\n")

    # 构建 Dataset
    dataset = PtDataloader(TRAIN_TXT, use_depth=USE_DEPTH)
    print(f"Dataset size: {len(dataset)} samples")

    # 构建 DataLoader（和训练时保持一致）
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,        # 跟你训练时保持一致
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ====== 可选：是否模拟把 batch 搬到 GPU ======
    # 如果你只想看「磁盘→CPU」的速度，设成 False
    # 如果你想看「磁盘→CPU→GPU」的整体速度，设成 True
    MOVE_TO_GPU = True
    # =============================================

    # 预热几个 batch（让 worker 启动起来）
    warmup_batches = 5
    print(f"Warmup {warmup_batches} batches...")
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        if MOVE_TO_GPU:
            if USE_DEPTH:
                rgb, depth, targets = batch
                rgb = rgb.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                rgb, targets = batch
                rgb = rgb.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

    torch.cuda.synchronize() if device.type == "cuda" else None

    # ====== 正式计时 ======
    print("\nStart timing DataLoader...")
    start_time = time.time()
    n_batches = 0
    n_samples = 0

    for batch in loader:
        if MOVE_TO_GPU:
            if USE_DEPTH:
                rgb, depth, targets = batch
                rgb = rgb.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                bs = rgb.size(0)
            else:
                rgb, targets = batch
                rgb = rgb.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                bs = rgb.size(0)
        else:
            # 只在 CPU 上统计 batch 大小
            if USE_DEPTH:
                rgb, depth, targets = batch
                bs = rgb.size(0)
            else:
                rgb, targets = batch
                bs = rgb.size(0)

        n_batches += 1
        n_samples += bs

        if n_batches >= N_TEST_BATCHES:
            break

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    elapsed = end_time - start_time

    it_per_sec = n_batches / elapsed
    samples_per_sec = n_samples / elapsed

    print("\n=== DataLoader Benchmark Result ===")
    print(f"Elapsed time : {elapsed:.2f} s")
    print(f"Batches      : {n_batches}")
    print(f"Samples      : {n_samples}")
    print(f"Iter / sec   : {it_per_sec:.2f} it/s")
    print(f"Samples / sec: {samples_per_sec:.2f} samples/s")
    print("===================================")


if __name__ == "__main__":
    main()
