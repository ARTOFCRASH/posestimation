import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ======== 配置区域 ========
SRC_ROOT = "/root/autodl-tmp/npz dataset"   # 你的原始压缩 npz 根目录
DST_ROOT = "/root/autodl-tmp/npz_dataset_cropped256"   # 新的输出目录
TARGET_SIZE = 256                                      # 输出尺寸
CROP_AND_RESIZE = True                                 # 是否按 depth 做 crop
# =========================


def crop_and_resize(color: np.ndarray,
                    depth: np.ndarray,
                    target_size: int = 256,
                    crop_and_resize: bool = True):
    """
    color: (H, W, 3), uint8
    depth: (H, W),    uint16 / float
    返回:
      color_t: [3, target_size, target_size] float32, 0~1
      depth_t: [1, target_size, target_size] float32
    """
    H, W, _ = color.shape

    # HWC -> CHW
    color_t = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0   # [3,H,W]
    depth_t = torch.from_numpy(depth).unsqueeze(0).float()               # [1,H,W]

    # ---------- 不做 crop，只整体 resize ----------
    if not crop_and_resize:
        color_t = F.interpolate(
            color_t.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        depth_t = F.interpolate(
            depth_t.unsqueeze(0),
            size=(target_size, target_size),
            mode="nearest"
        ).squeeze(0)

        return color_t, depth_t

    # ---------- 做 crop: 用 depth>0 当作前景 ----------
    mask = depth_t[0] > 0

    # 如果没有有效深度，退回整体 resize
    if not mask.any():
        color_t = F.interpolate(
            color_t.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        depth_t = F.interpolate(
            depth_t.unsqueeze(0),
            size=(target_size, target_size),
            mode="nearest"
        ).squeeze(0)

        return color_t, depth_t

    ys, xs = torch.where(mask)
    y_min, y_max = ys.min().item(), ys.max().item()
    x_min, x_max = xs.min().item(), xs.max().item()

    h = y_max - y_min + 1
    w = x_max - x_min + 1

    side = int(max(h, w) * 1.2)  # 加 20% margin
    if side < 1:
        side = max(h, w)

    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0

    y1 = int(round(cy - side / 2))
    y2 = y1 + side
    x1 = int(round(cx - side / 2))
    x2 = x1 + side

    # clip 到图像范围
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H, y2)
    x2 = min(W, x2)

    # ---- crop ----
    color_crop = color_t[:, y1:y2, x1:x2]
    depth_crop = depth_t[:, y1:y2, x1:x2]

    h_crop = color_crop.shape[1]
    w_crop = color_crop.shape[2]

    # 极端保护：防止空 crop
    if h_crop == 0 or w_crop == 0:
        color_crop = color_t
        depth_crop = depth_t
        h_crop, w_crop = H, W

    # ---- pad 成正方形 ----
    side2 = max(h_crop, w_crop)
    pad_h = side2 - h_crop
    pad_w = side2 - w_crop

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    color_pad = F.pad(
        color_crop,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0
    )
    depth_pad = F.pad(
        depth_crop,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0
    )

    # ---- resize 到 target_size x target_size ----
    color_resized = F.interpolate(
        color_pad.unsqueeze(0),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    depth_resized = F.interpolate(
        depth_pad.unsqueeze(0),
        size=(target_size, target_size),
        mode="nearest"
    ).squeeze(0)

    return color_resized, depth_resized


def main():
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        rel_dir = os.path.relpath(dirpath, SRC_ROOT)
        dst_dir = os.path.join(DST_ROOT, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        npz_files = [f for f in filenames if f.endswith(".npz")]
        if not npz_files:
            continue

        for fname in tqdm(npz_files, desc=rel_dir, leave=False):
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dst_dir, fname)

            # 已经处理过就跳过（方便中途中断、重跑）
            if os.path.exists(dst_path):
                continue

            data = np.load(src_path, allow_pickle=False)
            color = data["color"]     # (H,W,3)
            depth = data["depth"]     # (H,W)
            label = data["label"].astype(np.float32)   # (2,)

            color_t, depth_t = crop_and_resize(
                color,
                depth,
                target_size=TARGET_SIZE,
                crop_and_resize=CROP_AND_RESIZE
            )

            # 转回 numpy 存盘：
            # color: HWC uint8, depth: HW float32
            color_out = (
                color_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
            ).round().astype(np.uint8)

            depth_out = depth_t[0].cpu().numpy().astype(np.float16)

            # np.savez（不压缩）
            np.savez(
                dst_path,
                color=color_out,
                depth=depth_out,
                label=label,
            )


if __name__ == "__main__":
    main()
    print("Done. Converted compressed npz -> cropped + uncompressed npz.")
