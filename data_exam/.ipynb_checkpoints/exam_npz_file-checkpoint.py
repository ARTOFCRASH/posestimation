import os
import glob
import numpy as np
from collections import Counter

NPZ_DIR = r"/root/autodl-tmp/npz dataset/p98_m_npz"  # 这里改成你的 npz 文件夹
PRINT_FIRST_N = 5                      # 前多少个文件打印详细信息


def sizeof_fmt(num, suffix="B"):
    """把字节数转成人类可读形式，比如 14.3GB"""
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def main():
    npz_files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if not npz_files:
        print(f"[ERROR] No .npz files found in: {NPZ_DIR}")
        return

    print(f"Found {len(npz_files)} .npz files in {NPZ_DIR}")

    color_shapes = Counter()
    depth_shapes = Counter()
    label_shapes = Counter()

    color_dtypes = Counter()
    depth_dtypes = Counter()
    label_dtypes = Counter()

    total_color_bytes = 0
    total_depth_bytes = 0
    total_label_bytes = 0

    for idx, npz_path in enumerate(npz_files):
        try:
            # allow_pickle=False 更安全
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"[ERROR] Failed to load {npz_path}: {e}")
            continue

        keys = set(data.files)

        required_keys = {"color", "depth", "label"}
        if not required_keys.issubset(keys):
            print(f"[WARN] {os.path.basename(npz_path)} missing keys: {required_keys - keys}")
            continue

        color = data["color"]
        depth = data["depth"]
        label = data["label"]

        color_shapes[color.shape] += 1
        depth_shapes[depth.shape] += 1
        label_shapes[label.shape] += 1

        color_dtypes[color.dtype] += 1
        depth_dtypes[depth.dtype] += 1
        label_dtypes[label.dtype] += 1

        total_color_bytes += color.nbytes
        total_depth_bytes += depth.nbytes
        total_label_bytes += label.nbytes

        if idx < PRINT_FIRST_N:
            print("=" * 60)
            print(f"[{idx}] File: {os.path.basename(npz_path)}")
            print(f"  keys: {data.files}")
            print(f"  color: shape={color.shape}, dtype={color.dtype}, bytes={color.nbytes}")
            print(f"  depth: shape={depth.shape}, dtype={depth.dtype}, bytes={depth.nbytes}")
            print(f"  label: shape={label.shape}, dtype={label.dtype}, bytes={label.nbytes}")

        data.close()


    print("\n" + "#" * 60)
    print("SUMMARY")
    print("#" * 60)

    print(f"Total files: {len(npz_files)}")

    print("\n[COLOR] shapes:")
    for s, c in color_shapes.items():
        print(f"  shape={s}: count={c}")
    print("[COLOR] dtypes:")
    for dt, c in color_dtypes.items():
        print(f"  dtype={dt}: count={c}")

    print("\n[DEPTH] shapes:")
    for s, c in depth_shapes.items():
        print(f"  shape={s}: count={c}")
    print("[DEPTH] dtypes:")
    for dt, c in depth_dtypes.items():
        print(f"  dtype={dt}: count={c}")

    print("\n[LABEL] shapes:")
    for s, c in label_shapes.items():
        print(f"  shape={s}: count={c}")
    print("[LABEL] dtypes:")
    for dt, c in label_dtypes.items():
        print(f"  dtype={dt}: count={c}")

    total_bytes = total_color_bytes + total_depth_bytes + total_label_bytes

    print("\n[SIZE ESTIMATION]")
    print(f"  color total: {sizeof_fmt(total_color_bytes)}")
    print(f"  depth total: {sizeof_fmt(total_depth_bytes)}")
    print(f"  label total: {sizeof_fmt(total_label_bytes)}")
    print(f"  ALL total:   {sizeof_fmt(total_bytes)}")


if __name__ == "__main__":
    main()
