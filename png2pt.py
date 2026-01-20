import os
import re
import cv2
import torch
import math
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ======== 配置区域 ========
SRC_DIR = r"D:\files\persimmon data\RealSenseD405_raw\main"
DST_DIR = r"D:\files\persimmon data\RealSenseD405_raw\final_pt"
VIS_DIR = r"D:\files\persimmon data\RealSenseD405_raw\visual_check" # 可视化保存路径
TARGET_SIZE = 256
DELTA_MM = 60            # depth_filter 的保留范围
MARGIN_SCALE = 1.2          # bbox 扩大倍数
N_VIS_SAMPLES = 32          # 抽样多少个进行可视化检查
# =========================

os.makedirs(DST_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
PATTERN = re.compile(r"p(?P<num>\d+)_(?P<roll>-?\d+)_(?P<pitch>-?\d+)_color\.png")

def concat_color_depth_single(color_t, depth_t):
    """返回：PIL Image，左右拼接： [ COLOR | DEPTH ]"""
    color_np = (color_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_color = Image.fromarray(color_np)

    depth_np = depth_t[0].cpu().numpy()
    if depth_np.max() > depth_np.min():
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
    else:
        depth_norm = np.zeros_like(depth_np)
    
    depth_img = (depth_norm * 255).astype(np.uint8)
    img_depth = Image.fromarray(depth_img).convert("RGB")

    concat_img = Image.new("RGB", (img_color.width + img_depth.width, img_color.height))
    concat_img.paste(img_color, (0, 0))
    concat_img.paste(img_depth, (img_color.width, 0))
    return concat_img

def show_batch_grid(imgs_list, n_cols=4, save_path=None):
    """将拼接好的图列表转为 Grid"""
    if not imgs_list: return
    B = len(imgs_list)
    w_s, h_s = imgs_list[0].size
    n_rows = math.ceil(B / n_cols)
    grid_img = Image.new("RGB", (w_s * n_cols, h_s * n_rows))
    for idx, img in enumerate(imgs_list):
        grid_img.paste(img, ((idx % n_cols) * w_s, (idx // n_cols) * h_s))
    if save_path: grid_img.save(save_path)
    grid_img.show()

def keep_largest_component(mask_np):
    mask_u8 = mask_np.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1: return mask_np.astype(bool)
    max_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == max_id

def process_pipeline(color_path, depth_path):
    # 1. 读取并应用 depth_filter
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) # uint16 mm
    
    valid_mask = depth > 0
    if not valid_mask.any(): return None, None
    
    min_d = depth[valid_mask].min()
    mask_filter = (depth >= min_d) & (depth <= min_d + DELTA_MM) & (depth > 0)
    
    color[~mask_filter] = 0
    depth[~mask_filter] = 0
    
    # 2. Crop and Resize (256x256)
    H, W, _ = color.shape
    color_t = torch.from_numpy(color).permute(2, 0, 1).float() / 255.0
    depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)

    # 简单边界清理
    border = 15
    depth_t[:, :border, :] = 0; depth_t[:, -border:, :] = 0
    depth_t[:, :, :border] = 0; depth_t[:, :, -border:] = 0
    
    mask_np = keep_largest_component((depth_t[0] > 0).numpy())
    mask = torch.from_numpy(mask_np)
    
    if mask.sum() < 1000: # 面积太小则整图 Resize
        c_res = F.interpolate(color_t.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear').squeeze(0)
        d_res = F.interpolate(depth_t.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), mode='nearest').squeeze(0)
    else:
        ys, xs = torch.where(mask)
        y_min, y_max, x_min, x_max = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
        side = int(max(y_max-y_min, x_max-x_min) * MARGIN_SCALE)
        cy, cx = (y_min+y_max)/2.0, (x_min+x_max)/2.0
        
        y1, x1 = int(round(cy-side/2)), int(round(cx-side/2))
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(H, y1+side), min(W, x1+side)
        
        c_crop = color_t[:, y1:y2, x1:x2]
        d_crop = depth_t[:, y1:y2, x1:x2]
        
        side2 = max(c_crop.shape[1], c_crop.shape[2])
        pad_h, pad_w = side2 - c_crop.shape[1], side2 - c_crop.shape[2]
        c_pad = F.pad(c_crop, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
        d_pad = F.pad(d_crop, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
        
        c_res = F.interpolate(c_pad.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear').squeeze(0)
        d_res = F.interpolate(d_pad.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), mode='nearest').squeeze(0)
        
    return c_res, d_res

def main():
    color_files = sorted(Path(SRC_DIR).glob("*_color.png"))
    vis_imgs = []
    
    for c_path in tqdm(color_files, desc="Processing PNG -> PT"):
        m = PATTERN.match(c_path.name)
        if not m: continue
        
        d_path = c_path.with_name(c_path.name.replace("_color.png", "_depth.png"))
        if not d_path.exists(): continue
        
        c_res, d_res = process_pipeline(c_path, d_path)
        if c_res is None: continue
        
        # 保存为 .pt (使用 float16 节省空间)
        label = torch.tensor([float(m.group("roll")), float(m.group("pitch"))], dtype=torch.float32)
        save_data = {"color": c_res.half(), "depth": d_res.half(), "label": label}
        
        out_path = Path(DST_DIR) / c_path.name.replace("_color.png", ".pt")
        torch.save(save_data, out_path)
        
        # 收集可视化样本
        if len(vis_imgs) < N_VIS_SAMPLES:
            vis_imgs.append(concat_color_depth_single(c_res, d_res))

    # 展示并保存可视化结果
    show_batch_grid(vis_imgs, n_cols=4, save_path=os.path.join(VIS_DIR, f"batch{DELTA_MM}_check.png"))

if __name__ == "__main__":
    main()