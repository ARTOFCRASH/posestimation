import os
import numpy as np
import random
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import timm
from models import ResNet_CBAM, ResNet18_RGBD, ResNet18_RGB
from tqdm import tqdm
import torch.nn.init as init
import kornia.augmentation as K
import glob
from pt_dataloader import PtDataloader


#   设置种子
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


def angle_error_deg(pred, gt, eps=1e-8):
    # pred, gt: torch.Tensor, shape (B,3)
    pred = F.normalize(pred, dim=1, eps=eps)
    gt   = F.normalize(gt, dim=1, eps=eps)
    cos = (pred * gt).sum(dim=1).clamp(-1.0, 1.0)  # (B,)
    ang = torch.acos(cos) * (180.0 / torch.pi)        # (B,)
    return ang
'''
class DepthNormalize(object):

    valid depth min over dataset: 73
    valid depth max over dataset: 149
    mean depth range: 80.346625  ~  123.958125

    def __init__(self, max_depth=160.0):
        self.max_depth = max_depth

    def __call__(self, depth: torch.Tensor):
        # depth: [1, H, W], float
        depth = depth / self.max_depth
        depth = torch.clamp(depth, 0.0, 1.0)
        return depth
'''




# ====================== 读 txt 文件列表 ======================
def load_file_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":

    # ---------------------------- 超参数 ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    LR = 2e-5
    LR_DECAY = "0.1 every 20 epochs"
    NUM_EPOCHS = 80
    USE_DEPTH = True
    EARLY_STOP = 7
    model_name = "ResNet18_RGBD"
    PRE_TRAINED = True
    NUM_WORKERS = 8
    model = ResNet18_RGBD(pretrained=PRE_TRAINED, out_dim=3).to(device)
    current_time = time.strftime("%m%d%H%M", time.localtime())
    save_dir = f"/root/autodl-tmp/project/output/{model_name}/train8_vector_learning/"

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    # 目前 MyDataset 里默认是：permute + /255.0
    

    # ---------------------------- 数据增强 ----------------------------
    class BackgroundRandomizer:
        """针对合成数据黑背景的增强：将像素值为0的区域随机填充噪声或颜色"""
        def __init__(self, p=0.5):
            self.p = p
    
        def __call__(self, img): # img: [3, H, W] tensor
            if random.random() > self.p:
                return img
            # 找到黑色背景掩码 (假设合成数据背景为全黑)
            mask = (img.sum(dim=0, keepdim=True) < 0.01) # (Output) tensor(bool) [1, H, W]
            # 生成随机环境色或噪声
            noise = torch.rand_like(img) * 0.5 
            # 仅替换背景部分
            img = torch.where(mask, noise, img)
            return img


    train_color_transform = transforms.Compose([
    # 随机平移 (Translation)
    # degrees=0 不旋转
    # translate=(0.1, 0.1) 表示在宽和高方向最多平移 10% 的像素偏移
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    # 随机裁剪并缩放 (Random Resized Crop)
    # 模拟相机与柿子之间距离的变化（Scale）以及位置的不确定性
    # scale=(0.8, 1.0) 表示采样面积为原图的 80%~100%
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.95, 1.05)),

    # 极强的颜色与光照抖动 (Color Jittering)
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1)
    ], p=0.8),

    # 模拟真实相机的噪声与模糊
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.4),
    
    #背景随机化处理
    BackgroundRandomizer(p=0.7),
        
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_color_transform = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    
    '''    
    kornia_train_aug = torch.nn.Sequential(
    # K.ColorJitter(0.2, 0.2, 0.2, 0.02, p=1.0),
    # K.RandomGrayscale(p=0.1),
    K.Normalize(mean=imagenet_mean, std=imagenet_std),
    ).to(device)
    kornia_train_aug = None
    kornia_val_aug = K.Normalize(mean=imagenet_mean, std=imagenet_std).to(device)
    '''

    '''
    if USE_DEPTH:
        train_depth_transform = DepthNormalize(max_depth=160.0)
        val_depth_transform   = DepthNormalize(max_depth=160.0)
    else:
        train_depth_transform = None
        val_depth_transform   = None
    '''
    class DepthRandomize(object):
        def __init__(self, drop_prob=0.05, noise_std=2.0):
            """
            drop_prob: 随机孔洞概率
            noise_std: 高斯噪声标准差（单位 = 原始深度单位，如 mm）
            """
            self.drop_prob = drop_prob
            self.noise_std = noise_std
    
        def __call__(self, depth: torch.Tensor):
            # depth: [1,H,W], raw depth (e.g., mm), background = 0
    
            d = depth.clone()
            valid = d > 0
    
            if valid.any():
                # 随机孔洞
                drop = (torch.rand_like(d) < self.drop_prob) & valid
                d[drop] = 0.0
    
                # 高斯噪声
                noise = torch.randn_like(d) * float(self.noise_std)
                d[valid] = d[valid] + noise[valid]
    
                # 防止负深度
                d[valid] = torch.clamp(d[valid], min=0.0)
    
            return d

    
    class DepthBreakdown:
        """模拟 D405 真实相机的边缘破碎和内部空洞"""
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, depth): # depth: [1, H, W] tensor
            if random.random() > self.p:
                return depth
            d = depth.clone()
            # 1. 随机边缘侵蚀 (模拟 batch26 中的破碎边缘)
            mask = (d > 0).float()
            kernel_size = random.choice([3, 5])
            # 使用 MaxPool 模拟侵蚀效果
            eroded = -F.max_pool2d(-mask.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            d[eroded.squeeze(0) == 0] = 0
            
            # 2. 随机内部空洞 (模拟材质吸光导致的缺失)
            for _ in range(random.randint(1, 3)):
                h, w = d.shape[1], d.shape[2]
                cy, cx = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
                r = random.randint(5, 15)
                d[:, cy-r:cy+r, cx-r:cx+r] = 0
            return d


    class DepthNormalize(object):
        """
        per-image foreground standardization:
          D' = (D - mu_fg) / sigma_fg
        where mu_fg is median/mean on foreground pixels (depth>0),
        sigma_fg is MAD-based (robust) or std.
        Background stays 0.
        """
        def __init__(self, use_median=True, use_mad=True, clip=3.0, eps=1e-6):
            self.use_median = use_median   # True: median, False: mean
            self.use_mad = use_mad         # True: MAD, False: std
            self.clip = clip               # None or float
            self.eps = eps
    
        def __call__(self, depth: torch.Tensor):
            # depth: [1,H,W] float
            if depth.ndim != 3 or depth.size(0) != 1:
                raise ValueError(f"Depth must be [1,H,W], got {tuple(depth.shape)}")
    
            d = depth[0]
            mask = d > 0
    
            # no foreground -> return all zeros (or keep as is)
            if not mask.any():
                return torch.zeros_like(depth)
    
            vals = d[mask]
    
            # mu_fg
            mu = vals.median() if self.use_median else vals.mean()
    
            # sigma_fg
            if self.use_mad:
                mad = (vals - mu).abs().median()
                sigma = 1.4826 * mad  # robust std estimate
            else:
                sigma = vals.std(unbiased=False)
    
            sigma = torch.clamp(sigma, min=self.eps)
    
            out = depth.clone()
            out0 = out[0]
            out0[mask] = (out0[mask] - mu) / sigma
            out0[~mask] = 0.0
    
            if self.clip is not None:
                out = torch.clamp(out, -float(self.clip), float(self.clip))
    
            return out


    # pt数据集里的 depth是原始深度
    train_depth_transform = transforms.Compose([
        DepthBreakdown(p=0.6), # 模拟真实噪声
        DepthRandomize(drop_prob=0.1, noise_std=2.0),
        DepthNormalize(use_median=True, use_mad=True, clip=3.0)
    ])
    
    val_depth_transform = DepthNormalize(use_median=True, use_mad=True, clip=3.0)
    # ---------------------------- 数据集 ----------------------------
    '''
    train_shards = "/root/autodl-tmp/wds_kaki/train-{000000..000168}.tar"
    val_shards   = "/root/autodl-tmp/wds_kaki/val-{000000..000042}.tar"

    print(f"Using WebDataset shards:")
    print(f"  train: {train_shards}")
    print(f"  val:   {val_shards}")

    train_loader = get_wds_loader(
        train_shards,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        buffer_size=200,
        use_depth=USE_DEPTH,
    )

    val_loader = get_wds_loader(
        val_shards,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,     # 验证集不需要 shuffle
        buffer_size=0,
        use_depth=USE_DEPTH,
    )
    '''


    train_txt = "pt_train_files.txt"
    val_txt   = "pt_val_files.txt"
    
    train_dataset = PtDataloader(train_txt, use_depth=USE_DEPTH, color_transform=train_color_transform, depth_transform=train_depth_transform)
    val_dataset   = PtDataloader(val_txt, use_depth=USE_DEPTH, color_transform=val_color_transform, depth_transform=val_depth_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    # ---------------------------- 日志 & 保存目录 ----------------------------
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write(f"=== Training Log ===\n")
        f.write(f"Time: {current_time}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Pre-trained: {PRE_TRAINED}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"LR_DECAY: {LR_DECAY}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"USE_DEPTH: {USE_DEPTH}\n")
        f.write(f"EARLY_STOP: {EARLY_STOP}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n\n")
        
    writer = SummaryWriter(save_dir)
    best_model_path = os.path.join(save_dir, "best.pth")
    early_stopping = EarlyStopping(
        patience=EARLY_STOP,
        verbose=True,
        delta=0.0,
        path=best_model_path,   # 直接存到 best.pth
    )

    # ---------------------------- 模型 / 损失 / 优化器 ----------------------------

    def cosine_loss(pred, gt, eps=1e-8):
        pred = F.normalize(pred, dim=1, eps=eps)
        gt   = F.normalize(gt, dim=1, eps=eps)
        return 1 - F.cosine_similarity(pred, gt, dim=1, eps=eps).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    total_train_step = 0
    total_val_step = 0

    best_val_loss = float("inf")
    best_rmse = None
    best_std = None
    best_acc = None
    best_epoch = 0
    
    scaler = torch.amp.GradScaler("cuda")
    # ====================== 训练循环 ======================
    for epoch in range(NUM_EPOCHS):
        print(
            f"===================={model_name}: Epoch {epoch + 1}/{NUM_EPOCHS} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}=========================="
        )
        epoch_start = time.time()

        # -------------------- Train --------------------
        model.train()
        train_loss_epoch = 0.0
        train_samples = 0

        # WebDataset 没有固定长度，这里不指定 total
        for batch in tqdm(train_loader, total=len(train_loader)):
            if USE_DEPTH:
                rgb_inputs, depth_inputs, targets = batch
                rgb_inputs = rgb_inputs.to(device, non_blocking=True)
                depth_inputs = depth_inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                rgb_inputs, targets = batch
                rgb_inputs = rgb_inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                if USE_DEPTH:
                    outputs = model(rgb_inputs, depth_inputs)
                else:
                    outputs = model(rgb_inputs)
                    
                loss = cosine_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = rgb_inputs.size(0)
            train_loss_epoch += loss.item() * batch_size
            train_samples += batch_size

            total_train_step += 1
            if total_train_step % 100 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step=total_train_step)

        avg_train_loss = train_loss_epoch / max(1, train_samples)
        scheduler.step()

        # -------------------- Validation --------------------
        model.eval()
        val_loss_epoch = 0.0
        sum_squared_angle_error = 0.0
        sum_angle_error = 0.0
        all_angle_errors = []
        total_correct_angle = 0
        val_samples = 0
        angle_threshold = 3.0

        with torch.no_grad():
            for batch in val_loader:
                if USE_DEPTH:
                    rgb_inputs, depth_inputs, targets = batch
                    rgb_inputs = rgb_inputs.to(device)
                    depth_inputs = depth_inputs.to(device)
                    targets = targets.to(device)
                else:
                    rgb_inputs, targets = batch
                    rgb_inputs = rgb_inputs.to(device)
                    targets = targets.to(device)
                    
                with torch.amp.autocast("cuda"):
                    outputs = model(rgb_inputs, depth_inputs) if USE_DEPTH else model(rgb_inputs)
                    loss = cosine_loss(outputs, targets)

                batch_size = rgb_inputs.size(0)
                val_loss_epoch += loss.item() * batch_size
                val_samples += batch_size

                # 方向角度误差 (RMSE / std)
                angles = angle_error_deg(outputs, targets)  # (B,)
                sum_angle_error += angles.sum().item()
                sum_squared_angle_error += (angles ** 2).sum().item()
                all_angle_errors.extend(angles.cpu().tolist())

                total_correct_angle += (angles <= angle_threshold).sum().item()

        val_samples = max(1, val_samples)
        avg_val_loss = val_loss_epoch / val_samples
        val_acc = total_correct_angle / val_samples
        mean_angle = sum_angle_error / val_samples
        rmse_angle = np.sqrt(sum_squared_angle_error / val_samples)
        std_dev = np.std(all_angle_errors) if all_angle_errors else 0.0

        total_val_step += 1
        writer.add_scalar("Val/Loss", avg_val_loss, total_val_step)
        writer.add_scalar("Val/Mean", mean_angle, total_val_step)
        writer.add_scalar(f"Val/Error<={angle_threshold}", val_acc, total_val_step)
        writer.add_scalar("Val/RMSE", rmse_angle, total_val_step)
        writer.add_scalar("Val/SD", std_dev, total_val_step)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Mean Angle: {mean_angle:.3f} deg | "
            f"Angle RMSE: {rmse_angle:.3f} | "
            f"Angle std: {std_dev:.3f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        epoch_end = time.time()
        print(f"this epoch takes time: {epoch_end - epoch_start:.2f} s")

        # -------- 保存最佳模型（按 val loss，不是 checkpoint，只是 best.pth）--------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_rmse = rmse_angle
            best_std = std_dev
            best_acc = val_acc
            best_mean = mean_angle
            best_epoch = epoch + 1

            print(f"✅ Saved new best model at epoch {epoch + 1}, val loss={avg_val_loss:.4f}")
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("⏹ Early stopping triggered. Stop training.")
            break

    # --------- 写最终结果到日志 ----------
    with open(log_path, "a") as f:
        f.write(
            f"Best epoch = {best_epoch}, "
            f"best mean = {best_mean:.4f}, "
            f"best RMSE = {best_rmse:.4f}, "
            f"best std = {best_std:.4f}, "
            f"best Accuracy = {best_acc:.4f}\n"
        )

    writer.close()
    print("Training finished.")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")


