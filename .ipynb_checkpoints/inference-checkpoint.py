import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ResNet18_RGBD, ResNet18_RGB   
from pt_dataloader import PtDataloader


def rot_mat(point, vector, t):
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
    cos_theta = dot_product / (norm_a * norm_b + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    angle_in_degrees = np.degrees(theta)
    return angle_in_degrees


def rotate_result(roll, pitch):
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([1, 0, 0, 0])
    y_axis = np.array([0, 1, 0, 0])
    z_axis = np.array([0, 0, 1, 0])
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

# ===================== Depth transforms（跟你训练一致） =====================
class DepthRandomize(object):
    def __init__(self, drop_prob=0.05, noise_std=2.0):
        self.drop_prob = drop_prob
        self.noise_std = noise_std

    def __call__(self, depth: torch.Tensor):
        d = depth.clone()
        valid = d > 0
        if valid.any():
            drop = (torch.rand_like(d) < self.drop_prob) & valid
            d[drop] = 0.0
            noise = torch.randn_like(d) * float(self.noise_std)
            d[valid] = d[valid] + noise[valid]
            d[valid] = torch.clamp(d[valid], min=0.0)
        return d


class DepthNormalize(object):
    def __init__(self, use_median=True, use_mad=True, clip=3.0, eps=1e-6):
        self.use_median = use_median
        self.use_mad = use_mad
        self.clip = clip
        self.eps = eps

    def __call__(self, depth: torch.Tensor):
        if depth.ndim != 3 or depth.size(0) != 1:
            raise ValueError(f"Depth must be [1,H,W], got {tuple(depth.shape)}")

        d = depth[0]
        mask = d > 0
        if not mask.any():
            return torch.zeros_like(depth)

        vals = d[mask]
        mu = vals.median() if self.use_median else vals.mean()

        if self.use_mad:
            mad = (vals - mu).abs().median()
            sigma = 1.4826 * mad
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


class DepthOffset(object):
    def __init__(self, offset=100.0, clamp_min=0.0):
        self.offset = float(offset)
        self.clamp_min = clamp_min

    def __call__(self, depth: torch.Tensor):
        # depth: [1,H,W]
        if depth.ndim != 3 or depth.size(0) != 1:
            raise ValueError(f"Depth must be [1,H,W], got {tuple(depth.shape)}")

        d = depth.clone()
        valid = d > 0
        if valid.any():
            d[valid] = d[valid] - self.offset
            if self.clamp_min is not None:
                d[valid] = torch.clamp(d[valid], min=float(self.clamp_min))
        return d



def evaluate(model, loader, device, use_depth=True):
    loss_fn = nn.MSELoss().to(device)

    model.eval()
    val_loss_epoch = 0.0
    total_roll_diff = 0.0
    total_pitch_diff = 0.0
    sum_squared_angle_error = 0.0
    all_angle_errors = []
    total_correct_angle = 0
    val_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if use_depth:
                rgb_inputs, depth_inputs, targets = batch
                rgb_inputs = rgb_inputs.to(device, non_blocking=True)
                depth_inputs = depth_inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                rgb_inputs, targets = batch
                rgb_inputs = rgb_inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(rgb_inputs, depth_inputs) if use_depth else model(rgb_inputs)
                loss = loss_fn(outputs, targets)

            batch_size = rgb_inputs.size(0)
            val_loss_epoch += loss.item() * batch_size
            val_samples += batch_size

            roll_diff = torch.abs(outputs[:, 0] - targets[:, 0])
            pitch_diff = torch.abs(outputs[:, 1] - targets[:, 1])
            total_roll_diff += roll_diff.sum().item()
            total_pitch_diff += pitch_diff.sum().item()

            roll_pred = outputs[:, 0].detach().cpu().numpy()
            pitch_pred = outputs[:, 1].detach().cpu().numpy()
            roll_true = targets[:, 0].detach().cpu().numpy()
            pitch_true = targets[:, 1].detach().cpu().numpy()
            for rp, pp, rt, pt in zip(roll_pred, pitch_pred, roll_true, pitch_true):
                angle_error = directional_acc(rp, pp, rt, pt)
                sum_squared_angle_error += angle_error ** 2
                all_angle_errors.append(angle_error)
                if angle_error <= 10.0:
                    total_correct_angle += 1

    val_samples = max(1, val_samples)
    avg_val_loss = val_loss_epoch / val_samples
    avg_roll_diff = total_roll_diff / val_samples
    avg_pitch_diff = total_pitch_diff / val_samples
    val_acc = total_correct_angle / val_samples
    rmse_angle = np.sqrt(sum_squared_angle_error / val_samples)
    std_dev = np.std(all_angle_errors) if all_angle_errors else 0.0

    return {
        "val_loss": avg_val_loss,
        "roll_mae": avg_roll_diff,
        "pitch_mae": avg_pitch_diff,
        "dir_acc<= 10 deg": val_acc,
        "dir_rmse": rmse_angle,
        "dir_std": std_dev,
        "n_samples": val_samples
    }


def main():
    # ============ 你需要改的配置 ============
    USE_DEPTH = True
    PRE_TRAINED = False  # 评估时不需要预训练，直接加载权重
    BATCH_SIZE = 256
    NUM_WORKERS = 8

    best_model_path = r"/root/autodl-tmp/project/output/ResNet18_RGBD/train7/best.pth"
    val_root = r"pt_val_files.txt"
    # ======================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- transforms：一定要和训练时 val 一致 -------
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    val_color_transform = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    if USE_DEPTH:
        val_depth_transform = transforms.Compose([
            DepthNormalize(use_median=True, use_mad=True, clip=3.0)
        ])
    else:
        val_depth_transform = None

    # ------- dataset / loader -------
    val_dataset = PtDataloader(
        val_root,
        use_depth=USE_DEPTH,
        color_transform=val_color_transform,
        depth_transform=val_depth_transform
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

    # ------- model -------
    if USE_DEPTH:
        model = ResNet18_RGBD(pretrained=PRE_TRAINED, out_dim=2).to(device)
    else:
        model = ResNet18_RGB(pretrained=PRE_TRAINED, out_dim=2).to(device)

    # ------- load weights -------
    ckpt = torch.load(best_model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 兼容 DataParallel 保存的 "module.xxx"
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=True)

    # ------- evaluate -------
    metrics = evaluate(model, val_loader, device, use_depth=USE_DEPTH)

    print("\n==== Best Model Evaluation (same as val) ====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>16s}: {v:.6f}")
        else:
            print(f"{k:>16s}: {v}")

if __name__ == "__main__":
    main()
