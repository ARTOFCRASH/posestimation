import os
import re
import glob
import cv2
import numpy as np
from tqdm import tqdm



SRC_DIR = r"D:\files\C++projects\images generate\build\Debug\p92_m"
DST_DIR = r"D:\files\npz dataset\p92_m_npz"   # 输出 npz 的目录
os.makedirs(DST_DIR, exist_ok=True)

COLOR_SUFFIX = "_color.png"
DEPTH_SUFFIX = "_depth.png"

stem_pattern = re.compile(
    r'^(?P<prefix1>[^_]+)_(?P<prefix2>[^_]+)_(?P<roll>-?\d+(?:\.\d+)?)_(?P<pitch>-?\d+(?:\.\d+)?)_$'
)


# ------------ 扫描所有 color PNG ------------
color_files = glob.glob(os.path.join(SRC_DIR, f"*{COLOR_SUFFIX}"))

print(f"Found {len(color_files)} color images.")

for color_path in tqdm(color_files):
    fname = os.path.basename(color_path)

    if not fname.endswith(COLOR_SUFFIX):
        continue

    # 去掉 "_color.png" 得到干净的 stem，例如 "p1_-10_15"
    stem = fname[:-len(COLOR_SUFFIX)]   # 或者 fname.replace(COLOR_SUFFIX, "")
    m = stem_pattern.match(stem)
    if m is None:
        # 如果某些文件名不符合规则，你可以选择 raise / skip
        # 这里先打印出来方便你检查
        print(f"[WARN] filename does not match pattern: {fname}")
        continue

    roll = float(m.group("roll"))
    pitch = float(m.group("pitch"))

    # 找到对应的 depth 图像
    depth_fname = stem + DEPTH_SUFFIX
    depth_path = os.path.join(SRC_DIR, depth_fname)
    if not os.path.exists(depth_path):
        print(f"[WARN] depth file not found for: {fname}")
        continue

    # ------------ 读图像 ------------
    # color: BGR → RGB（建议训练用 RGB）
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color is None:
        print(f"[WARN] failed to read color image: {color_path}")
        continue
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    # depth: 16-bit / 原始格式
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"[WARN] failed to read depth image: {depth_path}")
        continue

    # ------------ 组装 label ------------
    # [roll, pitch]
    label = np.array([roll, pitch], dtype=np.float32)

    # ------------ 保存为 npz ------------
    # npz 文件名可以直接用 stem 保持一致，例如 "p1_-10_15.npz"
    npz_path = os.path.join(DST_DIR, stem + ".npz")

    # color/ depth 保持原始 dtype，后面在 Dataset 里再做 normalize
    np.savez(
        npz_path,
        color=color,
        depth=depth,
        label=label,
    )
