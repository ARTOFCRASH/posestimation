import os
import numpy as np
import random
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import timm
from models import ResNet_CBAM, ResNet18_RGBD, ResNet18_RGB
from tqdm import tqdm
import torch.nn.init as init
import kornia.augmentation as K
import glob
from wds_loader import get_wds_loader


#   设置种子
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


def rot_mat(point, vector, t):
    """
    生成一个绕任意轴旋转的 4x4 旋转矩阵。

    Params:
    - point: 旋转轴经过的点，4维numpy数组，例如 [a, b, c, 1]
    - vector: 旋转轴的方向向量，单位向量，4维numpy数组，例如 [u, v, w, 0]
    - t: 旋转角度，单位为弧度

    返回:
    - 4x4 transformation matrix
    """
    u, v, w, _ = vector
    a, b, c, _ = point
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    one_minus_cos_t = 1 - cos_t

    matrix = np.array([
        [
            u * u + (v * v + w * w) * cos_t,
            u * v * one_minus_cos_t - w * sin_t,
            u * w * one_minus_cos_t + v * sin_t,
            (a * (v * v + w * w) - u * (b * v + c * w)) * one_minus_cos_t + (b * w - c * v) * sin_t
        ],
        [
            u * v * one_minus_cos_t + w * sin_t,
            v * v + (u * u + w * w) * cos_t,
            v * w * one_minus_cos_t - u * sin_t,
            (b * (u * u + w * w) - v * (a * u + c * w)) * one_minus_cos_t + (c * u - a * w) * sin_t
        ],
        [
            u * w * one_minus_cos_t - v * sin_t,
            v * w * one_minus_cos_t + u * sin_t,
            w * w + (u * u + v * v) * cos_t,
            (c * (u * u + v * v) - w * (a * u + b * v)) * one_minus_cos_t + (a * v - b * u) * sin_t
        ],
        [0, 0, 0, 1]
    ])

    return matrix


def count_angle(vector1, vector2):
    vector1 = vector1[:3]
    vector2 = vector2[:3]
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm_a * norm_b)
    theta = np.arccos(cos_theta)
    angle_in_degrees = np.degrees(theta)

    return angle_in_degrees


def rotate_result(roll, pitch):
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([1, 0, 0, 0])  # 世界 x 方向
    y_axis = np.array([0, 1, 0, 0])  # 世界 y 方向
    z_axis = np.array([0, 0, 1, 0])  # 世界 z 方向
    rollval = np.radians(roll)
    pitchval = np.radians(pitch)
    roll_mat = rot_mat(origin, y_axis, rollval)
    result1 = roll_mat @ z_axis
    x_axis = roll_mat @ x_axis
    pitch_mat = rot_mat(origin, x_axis, pitchval)
    result2 = pitch_mat @ result1
    return result2


def directional_acc(roll_predicted, pitch_predicted, roll_label, pitch_label):
    pre = rotate_result(roll_predicted, pitch_predicted)
    label = rotate_result(roll_label, pitch_label)
    return count_angle(pre, label)


class DepthNormalize(object):
    '''
    valid depth min over dataset: 73
    valid depth max over dataset: 149
    mean depth range: 80.346625  ~  123.958125
    '''
    def __init__(self, max_depth=160.0):
        self.max_depth = max_depth

    def __call__(self, depth: torch.Tensor):
        # depth: [1, H, W], float
        depth = depth / self.max_depth
        depth = torch.clamp(depth, 0.0, 1.0)
        return depth


# ====================== 读 txt 文件列表 ======================
def load_file_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":

    # ---------------------------- 超参数 ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    LR = 5e-5
    NUM_EPOCHS = 80
    USE_DEPTH = True
    EARLY_STOP = 7
    model_name = "ResNet18_RGBD"
    model = ResNet18_RGBD(pretrained=True, out_dim=2).to(device)
    current_time = time.strftime("%m%d%H%M", time.localtime())
    save_dir = f"/root/autodl-tmp/project/output/{model_name}/train4/"

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    # 目前 MyDataset 里默认是：permute + /255.0

    kornia_train_aug = torch.nn.Sequential(
    K.ColorJitter(0.2, 0.2, 0.2, 0.02, p=1.0),
    K.RandomGrayscale(p=0.1),
    K.Normalize(mean=imagenet_mean, std=imagenet_std),
    ).to(device)

    kornia_val_aug = K.Normalize(mean=imagenet_mean, std=imagenet_std).to(device)

    if USE_DEPTH:
        train_depth_transform = DepthNormalize(max_depth=160.0)
        val_depth_transform   = DepthNormalize(max_depth=160.0)
    else:
        train_depth_transform = None
        val_depth_transform   = None

    # ---------------------------- 数据集 ----------------------------
    train_shards = "/root/autodl-tmp/wds_kaki/train-{000000..000168}.tar"
    val_shards   = "/root/autodl-tmp/wds_kaki/val-{000000..000042}.tar"

    print(f"Using WebDataset shards:")
    print(f"  train: {train_shards}")
    print(f"  val:   {val_shards}")

    train_loader = get_wds_loader(
        train_shards,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        buffer_size=1000,
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

    # ---------------------------- 日志 & 保存目录 ----------------------------
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write(f"=== Training Log ===\n")
        f.write(f"Time: {current_time}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"LR: {LR}\n")
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
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

    total_train_step = 0
    total_val_step = 0

    best_val_loss = float("inf")
    best_avg_roll_diff = None
    best_avg_pitch_diff = None
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
        for batch in tqdm(train_loader):
            if USE_DEPTH:
                rgb_inputs, depth_inputs, targets = batch
                rgb_inputs = rgb_inputs.to(device, non_blocking=True)
                depth_inputs = depth_inputs.to(device, non_blocking=True)
                # 在这里做 depth 归一化（原来是在 Dataset 里做的）
                if train_depth_transform is not None:
                    depth_inputs = train_depth_transform(depth_inputs)
                targets = targets.to(device, non_blocking=True)
            else:
                rgb_inputs, targets = batch
                rgb_inputs = rgb_inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                depth_inputs = None

            optimizer.zero_grad()
            rgb_inputs = kornia_train_aug(rgb_inputs)
            
            with torch.amp.autocast("cuda"):
                if USE_DEPTH:
                    outputs = model(rgb_inputs, depth_inputs)
                else:
                    outputs = model(rgb_inputs)
                    
                loss = loss_fn(outputs, targets)

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
        val_loss_epoch = 0.0
        total_roll_diff = 0.0
        total_pitch_diff = 0.0
        sum_squared_angle_error = 0.0
        all_angle_errors = []
        total_correct_angle = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if USE_DEPTH:
                    rgb_inputs, depth_inputs, targets = batch
                    rgb_inputs = rgb_inputs.to(device)
                    depth_inputs = depth_inputs.to(device)
                    if val_depth_transform is not None:
                        depth_inputs = val_depth_transform(depth_inputs)
                    targets = targets.to(device)
                else:
                    rgb_inputs, targets = batch
                    rgb_inputs = rgb_inputs.to(device)
                    targets = targets.to(device)
                    depth_inputs = None
                    
                rgb_inputs = kornia_val_aug(rgb_inputs)
                
                if USE_DEPTH:
                    outputs = model(rgb_inputs, depth_inputs)
                else:
                    outputs = model(rgb_inputs)
                    
                loss = loss_fn(outputs, targets)

                batch_size = rgb_inputs.size(0)
                val_loss_epoch += loss.item() * batch_size
                val_samples += batch_size

                # MAE (roll/pitch)
                roll_diff = torch.abs(outputs[:, 0] - targets[:, 0])
                pitch_diff = torch.abs(outputs[:, 1] - targets[:, 1])
                total_roll_diff += roll_diff.sum().item()
                total_pitch_diff += pitch_diff.sum().item()

                # 方向角度误差 (RMSE / std)
                roll_pred = outputs[:, 0].cpu().numpy()
                pitch_pred = outputs[:, 1].cpu().numpy()
                roll_true = targets[:, 0].cpu().numpy()
                pitch_true = targets[:, 1].cpu().numpy()
                for rp, pp, rt, pt in zip(roll_pred, pitch_pred, roll_true, pitch_true):
                    angle_error = directional_acc(rp, pp, rt, pt)
                    sum_squared_angle_error += angle_error ** 2
                    all_angle_errors.append(angle_error)
                    if angle_error <= 3.0:
                        total_correct_angle += 1

        val_samples = max(1, val_samples)
        avg_val_loss = val_loss_epoch / val_samples
        avg_roll_diff = total_roll_diff / val_samples
        avg_pitch_diff = total_pitch_diff / val_samples
        val_acc = total_correct_angle / val_samples
        rmse_angle = np.sqrt(sum_squared_angle_error / val_samples)
        std_dev = np.std(all_angle_errors) if all_angle_errors else 0.0

        total_val_step += 1
        writer.add_scalar("Val/Loss", avg_val_loss, total_val_step)
        writer.add_scalar("Val/Accuracy", val_acc, total_val_step)
        writer.add_scalars("Val/MAE", {
            "roll_MAE": avg_roll_diff,
            "pitch_MAE": avg_pitch_diff
        }, total_val_step)
        writer.add_scalar("Val/Directional_RMSE", rmse_angle, total_val_step)
        writer.add_scalar("Val/Directional_SD", std_dev, total_val_step)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Roll MAE: {avg_roll_diff:.4f} | Pitch MAE: {avg_pitch_diff:.4f} | "
            f"Angle RMSE: {rmse_angle:.4f} | Angle std: {std_dev:.4f}"
        )
        epoch_end = time.time()
        print(f"this epoch takes time: {epoch_end - epoch_start:.2f} s")

        # -------- 保存最佳模型（按 val loss，不是 checkpoint，只是 best.pth）--------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_avg_roll_diff = avg_roll_diff
            best_avg_pitch_diff = avg_pitch_diff
            best_rmse = rmse_angle
            best_std = std_dev
            best_acc = val_acc
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
            f"best MAE (roll) = {best_avg_roll_diff:.4f}, "
            f"best MAE (pitch) = {best_avg_pitch_diff:.4f}, "
            f"best RMSE = {best_rmse:.4f}, "
            f"best std = {best_std:.4f}, "
            f"best Accuracy = {best_acc:.4f}\n"
        )

    writer.close()
    print("Training finished.")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")


