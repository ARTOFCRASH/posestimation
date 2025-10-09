from torch.utils.data import ConcatDataset
from dataiter3channels import MyData
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
from localModels import ResNet34_CBAM
from tqdm import tqdm
import torch.nn.init as init


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


def create_concat_dataset(dir_list, transform=None, roll=False, pitch=False):
    datasets = []
    for d in dir_list:
        ds = MyData(img_dir=d, trans=transform, roll=roll, pitch=pitch)
        datasets.append(ds)
    # ConcatDataset 会将多个单独的 Dataset 级联起来
    concat_ds = ConcatDataset(datasets)
    return concat_ds


# 23个文件的绝对路径
all_dirs = []
for i in range(1, 24):
    folder_path = f"/root/autodl-tmp/croppedposeimages/data{i}"
    all_dirs.append(folder_path)

# random.shuffle(all_dirs)

folds = np.array_split(all_dirs, 5)  # 其中每个 folds[i] 是一个 numpy array，里面有4~5个文件夹路径
NUM_FOLDS = 5
all_fold_results = []  # 存放每折验证集的最终loss或别的指标

# ----------------------------Hyperparameters--------------------------------------------------------
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5699, 0.4200, 0.3462),
                         std=(0.3303, 0.2403, 0.2773))
])

BATCH_SIZE = 128
LR = 1e-4
NUM_EPOCHS = 100
# PATIENCE = 5  # 早停耐心
model_name = 'ResNet34_CBAM'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------------------------------------

current_time = time.strftime("%m%d%H%M", time.localtime())
save_dir = f'/root/autodl-tmp/results/{current_time}'
os.makedirs(save_dir, exist_ok=True)

# ========== 创建一个日志文件，写入超参数 ==========
log_path = os.path.join(save_dir, "train_log.txt")
with open(log_path, "w") as f:
    f.write(f"=== Training Log ===\n")
    f.write(f"Time: {current_time}\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"LR: {LR}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
    # f.write(f"PATIENCE: {PATIENCE}\n\n")

# 用于统计所有折的最佳指标 (MAE, acc等)，方便最后求平均
fold_best_mae_list = []
fold_best_acc_list = []

# 5折交叉验证
for fold_idx in range(NUM_FOLDS):

    fold_dir = os.path.join(save_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    best_model_path = os.path.join(fold_dir, "best.pth")
    # early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=best_model_path)

    val_dirs = folds[fold_idx]
    train_dirs = []
    for i in range(NUM_FOLDS):
        if i != fold_idx:
            train_dirs.extend(folds[i])

    print(f"\n===== Fold {fold_idx + 1}/{NUM_FOLDS} =====")
    print("Val set folders:", val_dirs)
    print("Train set folders:", train_dirs)

    train_dataset = create_concat_dataset(train_dirs, transform=transform, roll=True, pitch=True)  # roll or pitch?
    val_dataset = create_concat_dataset(val_dirs, transform=transform, roll=True, pitch=True)  # roll or pitch?
    train_data_size = len(train_dataset)
    test_data_size = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ResNet34_CBAM(2)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    writer = SummaryWriter(fold_dir)

    total_train_step = 0  # 记录训练次数
    total_test_step = 0  # 记录测试的次数
    # 初始化最优指标
    best_val_loss = float('inf')
    best_avg_roll_diff = None
    best_avg_pitch_diff = None
    best_test_acc = None
    best_epoch = 0

    # 训练
    for epoch in range(NUM_EPOCHS):
        print(
            f"===================={model_name}: {epoch + 1}th epoch started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}==========================")
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for rgb_inputs, depth_inputs, targets in tqdm(train_dataloader, total=len(train_dataloader)):
            rgb_inputs, depth_inputs, targets = rgb_inputs.to(device), depth_inputs.to(device), targets.to(device)
            output = model(rgb_inputs, depth_inputs)
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                writer.add_scalar("training loss", loss.item(), global_step=total_train_step)
            # 计算当前批次的总损失
            train_loss += loss.item() * rgb_inputs.size(0)

        # One epoch finished
        train_loss = train_loss / train_data_size
        # 学习率更新
        scheduler.step()

        # 验证
        total_val_accuracy = 0.0
        total_val_loss = 0.0
        model.eval()
        total_roll_diff = 0.0         # Roll mean absolute error
        total_pitch_diff = 0.0        # Pitch mean absolute error
        sum_squared_angle_error = 0.0    # directional root mean square error
        all_angle_errors = []           # standard deviation
        sample_count = 0
        with torch.no_grad():
            for rgb_inputs, depth_inputs, targets in val_dataloader:
                rgb_inputs, depth_inputs, targets = rgb_inputs.to(device), depth_inputs.to(device), targets.to(device)
                outputs = model(rgb_inputs, depth_inputs)
                loss = loss_fn(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                diff = torch.abs(outputs - targets)
                # 误差小于3°视为准确
                total_val_accuracy += torch.sum((diff[:, 0] <= 3) & (diff[:, 1] <= 3)).item()
                # 计算roll和pitch的差值
                roll_diff = torch.abs(outputs[:, 0] - targets[:, 0])
                pitch_diff = torch.abs(outputs[:, 1] - targets[:, 1])
                total_roll_diff += roll_diff.sum().item()
                total_pitch_diff += pitch_diff.sum().item()

                roll_pred = outputs[:, 0].cpu().numpy()
                pitch_pred = outputs[:, 1].cpu().numpy()
                roll_true = targets[:, 0].cpu().numpy()
                pitch_true = targets[:, 1].cpu().numpy()
                for rp, pp, rt, pt in zip(roll_pred, pitch_pred, roll_true, pitch_true):
                    angle_error = directional_acc(rp, pp, rt, pt)
                    sum_squared_angle_error += angle_error ** 2
                    all_angle_errors.append(angle_error)

        #  统计所有验证结果
        avg_val_loss = total_val_loss / test_data_size
        avg_roll_diff = total_roll_diff / test_data_size
        avg_pitch_diff = total_pitch_diff / test_data_size
        test_acc = total_val_accuracy / test_data_size
        std_dev = np.std(all_angle_errors) if all_angle_errors else 0.0
        rmse_angle = np.sqrt(sum_squared_angle_error / test_data_size)

        total_test_step += 1

        writer.add_scalar("validation Loss", avg_val_loss, total_test_step)
        writer.add_scalar("validation Accuracy", test_acc, total_test_step)
        writer.add_scalars("MAE", {'roll_MAE': avg_roll_diff, 'pitch_MAE': avg_pitch_diff}, total_test_step)
        writer.add_scalar("Directional RMSE", rmse_angle, total_test_step)
        writer.add_scalar("Directional SD", std_dev, total_test_step)

        print(
            f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {(total_val_loss / test_data_size):.4f}')
        end_time = time.time()
        print(f'this epoch takes time: {end_time - start_time}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_avg_roll_diff = avg_roll_diff
            best_avg_pitch_diff = avg_pitch_diff
            best_rmse = rmse_angle
            best_std = std_dev
            best_test_acc = test_acc
            best_epoch = epoch + 1  # 保存最佳的 epoch

        # early_stopping(total_val_loss / test_data_size, model)
        # if early_stopping.early_stop:
        #     print("early stopping")
        #     break

    # ==========  写入日志文件该折信息 ==========
    with open(log_path, "a") as f:
        f.write(f"Fold {fold_idx + 1} best epoch = {best_epoch}, "
                f"best MAE (roll) = {best_avg_roll_diff:.4f}, best MAE (pitch) = {best_avg_pitch_diff:.4f}, "
                f"best RMSE = {best_rmse:.4f}, "
                f"best std = {best_std:.4f}, "
                f"best Accuracy = {best_test_acc:.4f}\n")

    writer.close()

