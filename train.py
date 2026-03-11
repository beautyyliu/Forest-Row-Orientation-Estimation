"""
Training script for BEV-U-Net forest row orientation estimation.
"""

import os
import glob
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# =========================
# Configuration
# =========================
@dataclass
class Config:
    # Dataset paths
    pcd_dir = "val_data/val_0_120/pcd"
    json_dir = "val_data/val_0_120/json"

    # Output paths
    log_dir = "runs/supervised_learning"
    out_dir = "checkpoints/supervised_learning"

    # Reproducibility
    seed = 42

    # BEV settings
    bev_res = 0.2
    bev_x_range = (-10, 20)
    bev_y_range = (-10, 20)
    bev_clip_max_h = 10.0

    # Training settings
    batch_size = 8
    num_workers = 0
    lr = 1e-3
    weight_decay = 1e-5
    max_epochs = 200
    patience = 30
    val_ratio = 0.2

    # Loss weights
    cos_weight = 1.0
    tv_weight = 0.05

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# =========================
# Utility functions
# =========================
def set_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_relative_angle(yaw_deg, forest_deg=0.0):
    """Compute relative row orientation in [-90, 90]."""
    d = (forest_deg - yaw_deg + 180.0) % 360.0 - 180.0
    if d > 90.0:
        d -= 180.0
    if d < -90.0:
        d += 180.0
    return d


def deg2uv(angle_deg):
    """Convert angle (deg) to unit direction vector."""
    angle_rad = np.deg2rad(angle_deg)
    return np.cos(angle_rad), np.sin(angle_rad)


# =========================
# BEV construction
# =========================
def points_to_bev(points_xyz, x_range, y_range, res=0.2, clip_max_h=30.0):
    """Convert point cloud into a 3-channel BEV map."""
    H = int((y_range[1] - y_range[0]) / res)
    W = int((x_range[1] - x_range[0]) / res)

    if points_xyz.shape[0] == 0:
        return np.zeros((3, H, W), dtype=np.float32)

    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    z = np.clip(z, -clip_max_h, clip_max_h)

    mask = (
        (x >= x_range[0]) & (x < x_range[1]) &
        (y >= y_range[0]) & (y < y_range[1])
    )
    x, y, z = x[mask], y[mask], z[mask]

    xi = ((x - x_range[0]) / res).astype(np.int32)
    yi = ((y - y_range[0]) / res).astype(np.int32)

    density = np.zeros((H, W), dtype=np.float32)
    max_h = np.full((H, W), -1e9, dtype=np.float32)
    sum_h = np.zeros((H, W), dtype=np.float64)
    sum_h2 = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.int32)

    for gx, gy, gz in zip(xi, yi, z):
        if 0 <= gx < W and 0 <= gy < H:
            density[gy, gx] += 1.0
            max_h[gy, gx] = max(max_h[gy, gx], gz)
            sum_h[gy, gx] += gz
            sum_h2[gy, gx] += gz * gz
            count[gy, gx] += 1

    density = np.log1p(density)
    max_h[max_h < -1e8] = 0.0

    mean_h = np.zeros_like(max_h, dtype=np.float32)
    var_h = np.zeros_like(max_h, dtype=np.float32)

    valid = count > 0
    mean_h[valid] = (sum_h[valid] / count[valid]).astype(np.float32)
    var_h[valid] = (
        sum_h2[valid] / count[valid] - mean_h[valid] ** 2
    ).astype(np.float32)

    def normalize_positive_only(img):
        """Normalize positive values to [0, 1]."""
        out = np.zeros_like(img, dtype=np.float32)
        pos = img > 0
        if pos.sum() == 0:
            return out

        vals = img[pos].astype(np.float32)
        v1, v99 = np.percentile(vals, 1), np.percentile(vals, 99)

        if v99 - v1 < 1e-6:
            v1, v99 = vals.min(), vals.max()
        if v99 - v1 < 1e-6:
            out[pos] = 1.0
            return out

        out[pos] = np.clip((vals - v1) / (v99 - v1), 0.0, 1.0)
        return out

    density = normalize_positive_only(density)
    max_h = normalize_positive_only(max_h)
    var_h = normalize_positive_only(var_h)

    bev = np.stack([density, max_h, var_h], axis=0)
    return bev.astype(np.float32)


# =========================
# Dataset
# =========================
class ForestDirDataset(Dataset):
    """Dataset for forest row orientation estimation."""

    def __init__(self, pcd_dir, json_dir, cfg):
        self.pcd_dir = Path(pcd_dir)
        self.json_dir = Path(json_dir)
        self.cfg = cfg

        pcd_files = sorted(glob.glob(str(self.pcd_dir / "frame_*.pcd")))
        pairs = []

        for pcd_path in pcd_files:
            stem = Path(pcd_path).stem
            json_path = self.json_dir / f"{stem}.json"
            if json_path.exists():
                pairs.append((pcd_path, str(json_path)))

        if not pairs:
            raise FileNotFoundError("No matched (pcd, json) pairs found.")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _load_angle_uv(self, json_path):
        """Load orientation label from metadata."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        yaw_rad = data["uav_pose_current"][-1]

        # plantation orientation in world frame
        # fixed to 0 in the original dataset, randomized in the extended experiments
        forest_rad = 0.0
        forest_rad = data["env_yaw"]

        yaw_deg_raw = math.degrees(yaw_rad)
        forest_deg_raw = math.degrees(forest_rad)

        rel_angle_deg = normalize_relative_angle(
            yaw_deg=float(yaw_deg_raw),
            forest_deg=float(forest_deg_raw)
        )
        u, v = deg2uv(rel_angle_deg)
        return np.array([u, v], dtype=np.float32)

    def __getitem__(self, idx):
        pcd_path, json_path = self.pairs[idx]

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points, dtype=np.float32)

        bev = points_to_bev(
            pts,
            x_range=self.cfg.bev_x_range,
            y_range=self.cfg.bev_y_range,
            res=self.cfg.bev_res,
            clip_max_h=self.cfg.bev_clip_max_h
        )

        uv = self._load_angle_uv(json_path)
        H, W = bev.shape[1], bev.shape[2]

        uv_map = np.repeat(uv.reshape(2, 1, 1), repeats=H, axis=1)
        uv_map = np.repeat(uv_map, repeats=W, axis=2)

        return {
            "bev": torch.from_numpy(bev),
            "uv": torch.from_numpy(uv_map),
            "stem": Path(pcd_path).stem
        }


# =========================
# Model
# =========================
class DoubleConv(nn.Module):
    """Two-layer conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """U-Net for dense orientation prediction."""

    def __init__(self, in_ch=3, out_ch=2, base=32):
        super().__init__()

        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)

        self.mid = DoubleConv(base * 4, base * 8)

        self.up3_conv = DoubleConv(base * 8 + base * 4, base * 4)
        self.up2_conv = DoubleConv(base * 4 + base * 2, base * 2)
        self.up1_conv = DoubleConv(base * 2 + base, base)

        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    @staticmethod
    def _upsample_to(x, ref):
        """Upsample to target size."""
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m = self.mid(self.pool3(d3))

        u3 = self._upsample_to(m, d3)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3_conv(u3)

        u2 = self._upsample_to(u3, d2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2_conv(u2)

        u1 = self._upsample_to(u2, d1)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1_conv(u1)

        return self.out(u1)


# =========================
# Loss functions
# =========================
def cosine_direction_loss(pred_uv, gt_uv, eps=1e-6):
    """Sign-invariant cosine loss."""
    pred_norm = torch.clamp(torch.sqrt((pred_uv ** 2).sum(dim=1, keepdim=True)), min=eps)
    gt_norm = torch.clamp(torch.sqrt((gt_uv ** 2).sum(dim=1, keepdim=True)), min=eps)

    pred_u = pred_uv / pred_norm
    gt_u = gt_uv / gt_norm

    cos = (pred_u * gt_u).sum(dim=1, keepdim=True)
    return (1.0 - cos ** 2).mean()


def tv_smooth_loss(pred_uv):
    """Total variation smoothness loss."""
    du_dx = torch.abs(pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1]).mean()
    du_dy = torch.abs(pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :]).mean()
    return du_dx + du_dy


# =========================
# Training helpers
# =========================
class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True


def train_val_split(dataset, val_ratio=0.2):
    """Split dataset into train and validation sets."""
    n = len(dataset)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val
    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(CFG.seed)
    )


def save_model(model, path):
    """Save model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# =========================
# Main training loop
# =========================
def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.out_dir, exist_ok=True)

    dataset = ForestDirDataset(CFG.pcd_dir, CFG.json_dir, CFG)
    train_set, val_set = train_val_split(dataset, CFG.val_ratio)

    train_loader = DataLoader(
        train_set,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    model = UNetSmall(in_ch=3, out_ch=2, base=32).to(CFG.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        verbose=True
    )

    writer = SummaryWriter(CFG.log_dir)
    early = EarlyStopping(patience=CFG.patience, min_delta=1e-5)

    best_val = float("inf")
    best_model_path = os.path.join(CFG.out_dir, "best_model.pth")
    last_model_path = os.path.join(CFG.out_dir, "last_model.pth")
    global_step = 0

    print(f"Device: {CFG.device}")
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        train_loss_meter = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.max_epochs} [Train]")
        for batch in pbar:
            bev = batch["bev"].to(CFG.device)
            uv = batch["uv"].to(CFG.device)

            pred = model(bev)
            loss_cos = cosine_direction_loss(pred, uv)
            loss_tv = tv_smooth_loss(pred)
            loss = CFG.cos_weight * loss_cos + CFG.tv_weight * loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/loss_cos", loss_cos.item(), global_step)
            writer.add_scalar("train/loss_tv", loss_tv.item(), global_step)
            global_step += 1

        train_loss_epoch = train_loss_meter / max(1, len(train_loader))

        model.eval()
        val_loss_meter = 0.0

        with torch.no_grad():
            for batch in val_loader:
                bev = batch["bev"].to(CFG.device)
                uv = batch["uv"].to(CFG.device)

                pred = model(bev)
                loss_cos = cosine_direction_loss(pred, uv)
                loss_tv = tv_smooth_loss(pred)
                loss = CFG.cos_weight * loss_cos + CFG.tv_weight * loss_tv

                val_loss_meter += loss.item()

        val_loss_epoch = val_loss_meter / max(1, len(val_loader))

        writer.add_scalar("val/loss", val_loss_epoch, epoch)
        writer.add_scalar("train/epoch_loss", train_loss_epoch, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_loss_epoch)

        if val_loss_epoch < best_val:
            best_val = val_loss_epoch
            save_model(model, best_model_path)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss_epoch:.6f} | "
            f"val_loss={val_loss_epoch:.6f} | "
            f"best_val={best_val:.6f}"
        )

        early.step(val_loss_epoch)
        if early.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    save_model(model, last_model_path)
    writer.close()

    print(f"Best model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")
    print(f"TensorBoard logdir: {CFG.log_dir}")


if __name__ == "__main__":
    main()