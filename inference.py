import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from models import ResNet18_RGBD, ResNet18_RGB   
from pt_dataloader import PtDataloader


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


def cosine_loss(pred, gt, eps=1e-8):
    pred = F.normalize(pred, dim=1, eps=eps)
    gt   = F.normalize(gt, dim=1, eps=eps)
    return 1 - F.cosine_similarity(pred, gt, dim=1, eps=eps).mean()


def angle_error_deg(pred, gt, eps=1e-8):
    # pred, gt: torch.Tensor, shape (B,3)
    pred = F.normalize(pred, dim=1, eps=eps)
    gt   = F.normalize(gt, dim=1, eps=eps)
    cos = (pred * gt).sum(dim=1).clamp(-1.0, 1.0)     # (B,)
    ang = torch.acos(cos) * (180.0 / torch.pi)        # (B,)
    return ang

    
def evaluate(model, loader, device, use_depth=True, angle_threshold=3.0):
    model.eval()

    val_loss_sum = 0.0
    val_samples = 0

    sum_angle = 0.0
    sum_sq_angle = 0.0
    all_angles = []
    total_correct = 0

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
                loss = cosine_loss(outputs, targets)

            B = targets.size(0)
            val_loss_sum += loss.item() * B
            val_samples += B

            angles = angle_error_deg(outputs, targets)  # (B,)
            sum_angle += angles.sum().item()
            sum_sq_angle += (angles ** 2).sum().item()
            all_angles.extend(angles.detach().cpu().tolist())
            total_correct += (angles <= angle_threshold).sum().item()

    val_samples = max(1, val_samples)
    avg_loss = val_loss_sum / val_samples
    mean_angle = sum_angle / val_samples
    rmse_angle = np.sqrt(sum_sq_angle / val_samples)
    std_angle = float(np.std(all_angles)) if all_angles else 0.0
    acc = total_correct / val_samples

    return {
        "summary"：f"Test result on {val_samples} samples",
        "val_loss(cos)": avg_loss,
        "mean": mean_angle,
        "rmse": rmse_angle,
        "std": std_angle,
        f"acc@{angle_threshold:.1f}deg": acc,
    }


def main():
    # ============ 你需要改的配置 ============
    USE_DEPTH = True
    PRE_TRAINED = False  # 评估时不需要预训练，直接加载权重
    BATCH_SIZE = 256
    NUM_WORKERS = 8

    best_model_path = r"D:\files\projects\output\ResNet18_RGBD\train7\best.pth"
    val_root = r"D:\files\persimmon data\RealSenseD405_raw\final_pt"
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
        model = ResNet18_RGBD(pretrained=PRE_TRAINED, out_dim=3).to(device)
    else:
        model = ResNet18_RGB(pretrained=PRE_TRAINED, out_dim=3).to(device)

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
    metrics = evaluate(model, val_loader, device, use_depth=USE_DEPTH,, angle_threshold=3.0)
    print("\n==== Inference result ====")
    print(metrics["summary"])
    print(f"loss : {metrics['val_loss(cos)']:.6f}")
    print(f"mean : {metrics['mean']:.3f} deg")
    print(f"rmse : {metrics['rmse']:.3f} deg")
    print(f"std  : {metrics['std']:.3f} deg")
    for k in metrics:
        if k.startswith("acc@"):
            print(f"{k}: {metrics[k]:.4f}")

if __name__ == "__main__":
    main()
