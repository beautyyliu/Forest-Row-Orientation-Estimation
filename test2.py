"""
Test script for evaluating the BEV-U-Net model under varying plantation orientations.

This script corresponds to the experiments reported in:
    - Section 5.4.1: Randomized Plantation Orientation
    - Section 5.4.2: Unseen Orientation Generalization

The code itself does not need to be modified. To run different experiments,
only the following paths need to be updated:
    - pcd_dir
    - json_dir
    - ckpt_path

Experiment settings:

Section 5.4.1 — Randomized Plantation Orientation
Training plantation orientation range: (-90°, 90°)
Testing plantation orientation range:  (-90°, 90°)
    pcd_dir = "val_data/val_0_180/pcd"
    json_dir = "val_data/val_0_180/json"
    ckpt_path = "model/2_model_with_forest_yaw.pth"

Section 5.4.2 — Unseen Orientation Generalization
Seen orientations
Training plantation orientation range: (-30°, 90°)
Testing plantation orientation range:  (-30°, 90°)
    pcd_dir = "val_data/val_0_120/pcd"
    json_dir = "val_data/val_0_120/json"
    ckpt_path = "model/3_model_yaw_0-120.pth"

Unseen orientations
Training plantation orientation range: (-30°, 90°)
Testing plantation orientation range:  (-90°, -30°)
    pcd_dir = "val_data/val_120_180/pcd"
    json_dir = "val_data/val_120_180/json"
    ckpt_path = "model/3_model_yaw_0-120.pth"
"""

import os
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# Configuration
# =========================
class TestCfg:
    pcd_dir = "val_data/val_120_180/pcd"
    json_dir = "val_data/val_120_180/json"
    ckpt_path = "model/3_model_yaw_0-120.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    num_workers = 0

    save_vis = False
    vis_out_dir = "./val_vis"
    vis_every_n = 1

    bev_res = 0.2
    bev_x_range = (-10, 20)
    bev_y_range = (-10, 20)
    bev_clip_max_h = 10.0


# =========================
# Utilities
# =========================
def adjust_angle(yaw_deg, forest_deg=0.0):
    """Compute relative row orientation in [-90, 90]."""
    d = (forest_deg - yaw_deg + 180.0) % 360.0 - 180.0
    if d > 90.0:
        d -= 180.0
    if d < -90.0:
        d += 180.0
    return d


def deg2uv(deg):
    """Convert angle (deg) to unit direction vector."""
    rad = np.deg2rad(deg)
    return np.cos(rad), np.sin(rad)


def norm_pos_only(img):
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

    out[pos] = np.clip((vals - v1) / (v99 - v1), 0, 1)
    return out


def points_to_bev(points_xyz, x_range, y_range, res=0.2, clip_max_h=10.0):
    """Convert point cloud into a 3-channel BEV map."""
    if points_xyz.shape[0] == 0:
        H = int((y_range[1] - y_range[0]) / res)
        W = int((x_range[1] - x_range[0]) / res)
        return np.zeros((3, H, W), dtype=np.float32)

    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    z = np.clip(z, -clip_max_h, clip_max_h)

    mask = (
        (x >= x_range[0]) & (x < x_range[1]) &
        (y >= y_range[0]) & (y < y_range[1])
    )
    x, y, z = x[mask], y[mask], z[mask]

    W = int((x_range[1] - x_range[0]) / res)
    H = int((y_range[1] - y_range[0]) / res)

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

    if count.max() > 0:
        density = (count.astype(np.float32) / count.max()).astype(np.float32)
    else:
        density = np.zeros_like(count, dtype=np.float32)

    max_h[max_h < -1e8] = 0.0

    mean_h = np.zeros_like(max_h, dtype=np.float32)
    nz = count > 0
    mean_h[nz] = (sum_h[nz] / count[nz]).astype(np.float32)

    var_h = np.zeros_like(max_h, dtype=np.float32)
    var_h[nz] = (sum_h2[nz] / count[nz] - mean_h[nz] ** 2).astype(np.float32)

    max_h = norm_pos_only(max_h)
    var_h = norm_pos_only(var_h)

    bev = np.stack([density, max_h, var_h], axis=0).astype(np.float32)
    return bev


def build_center_mask(H, W, x_range, y_range, center_half_span=10.0):
    """Build mask for the center region."""
    xs = np.linspace(x_range[0], x_range[1], W, endpoint=False)
    ys = np.linspace(y_range[0], y_range[1], H, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)
    return (
        (xv >= -center_half_span) & (xv <= center_half_span) &
        (yv >= -center_half_span) & (yv <= center_half_span)
    )


def angle_diff_180_deg(a_deg, b_deg):
    """Compute minimum angle difference under 180-degree symmetry."""
    d = ((a_deg - b_deg + 90.0) % 180.0) - 90.0
    return abs(d)


def prop_lt(arr, th):
    """Compute proportion below threshold."""
    a = np.array(arr, dtype=np.float32)
    return float((a < th).mean())


# =========================
# Model
# =========================
class DoubleConv(nn.Module):
    """Two-layer conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """Lightweight U-Net for dense orientation prediction."""

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

        self.out = nn.Conv2d(base, out_ch, 1)

    def upsample_to(self, x, ref):
        """Upsample to target size."""
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m = self.mid(self.pool3(d3))

        u3 = self.upsample_to(m, d3)
        u3 = self.up3_conv(torch.cat([u3, d3], dim=1))

        u2 = self.upsample_to(u3, d2)
        u2 = self.up2_conv(torch.cat([u2, d2], dim=1))

        u1 = self.upsample_to(u2, d1)
        u1 = self.up1_conv(torch.cat([u1, d1], dim=1))

        return self.out(u1)


# =========================
# Dataset
# =========================
class ForestDirVal(Dataset):
    """Validation dataset for forest row orientation estimation."""

    def __init__(self, pcd_dir, json_dir, cfg):
        self.pcd_dir = Path(pcd_dir)
        self.json_dir = Path(json_dir)
        self.cfg = cfg

        pcd_files = sorted(glob.glob(str(self.pcd_dir / "frame_*.pcd")))
        self.items = []

        for pcd_path in pcd_files:
            stem = Path(pcd_path).stem
            json_path = self.json_dir / f"{stem}.json"
            self.items.append((pcd_path, str(json_path), json_path.exists()))

        if not self.items:
            raise FileNotFoundError(f"No PCDs found in {self.pcd_dir}")

    def __len__(self):
        return len(self.items)

    def load_angle_uv_map(self, json_path, H, W):
        """Load frame-level orientation label and expand it to a dense map."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        yaw_rad = float(data["uav_pose_current"][-1])

        # plantation orientation in world frame
        # fixed to 0 in the original dataset, randomized in the extended experiments
        forest_rad = data["env_yaw"]

        yaw_deg_raw = math.degrees(yaw_rad)
        forest_deg_raw = math.degrees(forest_rad)

        rel_deg = adjust_angle(float(yaw_deg_raw), float(forest_deg_raw))
        u, v = deg2uv(rel_deg)

        uv = np.array([u, v], dtype=np.float32).reshape(2, 1, 1)
        uv = np.repeat(uv, H, axis=1)
        uv = np.repeat(uv, W, axis=2)
        return uv

    def __getitem__(self, idx):
        pcd_path, json_path, has_label = self.items[idx]

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points, dtype=np.float32)

        t0 = time.perf_counter()
        bev = points_to_bev(
            pts,
            TestCfg.bev_x_range,
            TestCfg.bev_y_range,
            res=TestCfg.bev_res,
            clip_max_h=TestCfg.bev_clip_max_h
        )
        bev_time = time.perf_counter() - t0

        H, W = bev.shape[1], bev.shape[2]

        sample = {
            "bev": torch.from_numpy(bev),
            "stem": Path(pcd_path).stem,
            "has_label": has_label,
            "bev_time": float(bev_time),
        }

        if has_label:
            uv = self.load_angle_uv_map(json_path, H, W)
            sample["uv"] = torch.from_numpy(uv)

        return sample


# =========================
# Metrics
# =========================
def cosine_direction_loss(pred_uv, gt_uv, eps=1e-6):
    """Sign-invariant cosine loss."""
    pred_norm = torch.clamp(torch.sqrt((pred_uv ** 2).sum(dim=1, keepdim=True)), min=eps)
    gt_norm = torch.clamp(torch.sqrt((gt_uv ** 2).sum(dim=1, keepdim=True)), min=eps)

    pred_u = pred_uv / pred_norm
    gt_u = gt_uv / gt_norm

    cos = (pred_u * gt_u).sum(dim=1, keepdim=True)
    return (1.0 - cos ** 2).mean()


def angle_mae_deg(pred_uv, gt_uv):
    """Compute angular MAE and median error."""
    pred_u, pred_v = pred_uv[:, 0], pred_uv[:, 1]
    gt_u, gt_v = gt_uv[:, 0], gt_uv[:, 1]

    th_p = torch.atan2(pred_v, pred_u)
    th_g = torch.atan2(gt_v, gt_u)

    d = th_p - th_g
    d = (d + math.pi) % (2 * math.pi) - math.pi

    mae = torch.rad2deg(d.abs()).mean().item()
    med = torch.rad2deg(d.abs()).median().item()
    return mae, med


# =========================
# Main
# =========================
def main():
    ds = ForestDirVal(TestCfg.pcd_dir, TestCfg.json_dir, TestCfg)
    dl = DataLoader(
        ds,
        batch_size=TestCfg.batch_size,
        shuffle=False,
        num_workers=TestCfg.num_workers,
        pin_memory=False
    )

    model = UNetSmall(in_ch=3, out_ch=2, base=32).to(TestCfg.device)
    state = torch.load(TestCfg.ckpt_path, map_location=TestCfg.device)
    model.load_state_dict(state)
    model.eval()

    if TestCfg.save_vis:
        os.makedirs(TestCfg.vis_out_dir, exist_ok=True)

    have_label = 0
    cos_sum = 0.0
    all_frame_errs = []
    pred_frame_degs = []
    gt_frame_degs = []

    bin_step = 0.5
    bins = np.arange(0.0, 180.0 + bin_step, bin_step)
    hist_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total_pixels = 0

    pixel_mae_full_list = []
    pixel_mae_center_list = []
    frame_err_list = []

    total_pred_time = 0.0
    total_bev_time = 0.0
    count_samples = 0

    pbar = tqdm(total=len(dl), desc="Evaluating")

    for i, batch in enumerate(dl):
        bev = batch["bev"].to(TestCfg.device)
        stem = batch["stem"][0]

        bt = batch.get("bev_time", None)
        if bt is not None:
            if isinstance(bt, torch.Tensor):
                total_bev_time += bt.sum().item()
                count_samples += bt.numel()
            elif isinstance(bt, (list, tuple)):
                total_bev_time += float(np.sum(bt))
                count_samples += len(bt)
            else:
                total_bev_time += float(bt)
                count_samples += 1

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(bev)
        t1 = time.perf_counter()
        total_pred_time += (t1 - t0)

        if batch["has_label"][0]:
            have_label += 1
            uv = batch["uv"].to(TestCfg.device)

            loss_cos = cosine_direction_loss(pred, uv).item()
            cos_sum += loss_cos

            p_np = pred[0].detach().cpu().numpy()
            u = p_np[0]
            v = p_np[1]

            norm = np.sqrt(u * u + v * v) + 1e-6
            u = u / norm
            v = v / norm

            theta_deg = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
            theta180_map = theta_deg % 180.0

            H, W = u.shape
            mask_center = build_center_mask(
                H, W,
                TestCfg.bev_x_range,
                TestCfg.bev_y_range,
                center_half_span=10.0
            )

            dens = bev[0, 0].detach().cpu().numpy()
            mask_valid = dens > 0

            gt_np = uv[0].detach().cpu().numpy()
            gu = float(gt_np[0].mean())
            gv = float(gt_np[1].mean())
            gt_deg_full = (math.degrees(math.atan2(gv, gu)) + 360.0) % 360.0
            gt_theta180 = gt_deg_full % 180.0

            diffs_full = np.abs(theta180_map - gt_theta180)
            diffs_full = np.minimum(diffs_full, 180.0 - diffs_full)

            if mask_valid.any():
                pixel_mae_full = float(diffs_full[mask_valid].mean())
            else:
                pixel_mae_full = float(diffs_full.mean())
            pixel_mae_full_list.append(pixel_mae_full)

            mask_center_valid = mask_center & mask_valid
            if mask_center_valid.any():
                pixel_mae_center = float(diffs_full[mask_center_valid].mean())
            else:
                pixel_mae_center = float(
                    diffs_full[mask_center].mean() if mask_center.any() else diffs_full.mean()
                )
            pixel_mae_center_list.append(pixel_mae_center)

            angles_c = theta180_map[mask_center & mask_valid].ravel()
            if angles_c.size == 0:
                angles_c = theta180_map[mask_valid].ravel()
                if angles_c.size == 0:
                    angles_c = theta180_map.ravel()

            bin_step = 1.0
            bins = np.arange(0.0, 180.0 + bin_step, bin_step)
            hist_c, _ = np.histogram(angles_c, bins=bins)
            centers = (bins[:-1] + bins[1:]) * 0.5
            peak_idx_c = int(np.argmax(hist_c))
            peak_center_c = centers[peak_idx_c]

            mask_peak_c = (angles_c >= peak_center_c - 10.0) & (angles_c <= peak_center_c + 10.0)
            if mask_peak_c.sum() > 0:
                pred_deg_center = float(np.median(angles_c[mask_peak_c]))
            else:
                pred_deg_center = float(peak_center_c)

            diff_c = abs(pred_deg_center - gt_theta180)
            frame_err = float(min(diff_c, 180.0 - diff_c))
            frame_err_list.append(frame_err)

            all_frame_errs.append(frame_err)
            pred_frame_degs.append(pred_deg_center)
            gt_frame_degs.append(gt_theta180)

            if mask_valid.any():
                valid_errs = diffs_full[mask_valid].ravel()
                hist_add, _ = np.histogram(valid_errs, bins=bins)
                hist_counts[:len(hist_add)] += hist_add
                total_pixels += valid_errs.size

        if TestCfg.save_vis and (i % TestCfg.vis_every_n == 0):
            p_np = pred[0].detach().cpu().numpy()
            uu = p_np[0]
            vv = p_np[1]

            norm = np.sqrt(uu * uu + vv * vv) + 1e-6
            uu = uu / norm
            vv = vv / norm

            H, W = uu.shape
            step = max(1, min(H, W) // 5)

            dens = bev[0, 0].detach().cpu().numpy()
            nz = dens > 0
            vis = np.zeros_like(dens, dtype=np.float32)
            if nz.any():
                v1, v99 = np.percentile(dens[nz], 1), np.percentile(dens[nz], 99)
                if v99 - v1 < 1e-6:
                    v1, v99 = dens[nz].min(), dens[nz].max()
                if v99 - v1 < 1e-6:
                    vis[nz] = 1.0
                else:
                    vis[nz] = np.clip((dens[nz] - v1) / (v99 - v1), 0, 1)
            vis = np.power(vis, 0.5)

            Y, X = np.mgrid[0:H:step, 0:W:step]
            uu_s = uu[::step, ::step]
            vv_s = vv[::step, ::step]

            plt.figure(figsize=(6, 6))
            plt.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
            plt.quiver(
                X, Y, uu_s, vv_s,
                scale_units="xy", scale=0.1,
                width=0.004, headwidth=5, headlength=7,
                color="deepskyblue", alpha=0.9
            )

            if batch["has_label"][0]:
                gt_uv = batch["uv"][0].cpu().numpy()
                gu_s = gt_uv[0, ::step, ::step]
                gv_s = gt_uv[1, ::step, ::step]

                plt.quiver(
                    X, Y, gu_s, gv_s,
                    scale_units="xy", scale=0.05,
                    width=0.004, headwidth=5, headlength=7,
                    color="lime", alpha=0.6, label="GT"
                )

                plt.title(
                    f"{stem}\n"
                    f"pixelMAE(full)≈{pixel_mae_full_list[-1]:.2f}° | "
                    f"pixelMAE(center)≈{pixel_mae_center_list[-1]:.2f}° | "
                    f"frameErr(center-peak)≈{frame_err_list[-1]:.2f}°"
                )
            else:
                plt.title(f"{stem}")

            plt.axis("off")
            plt.tight_layout()
            out_path = os.path.join(TestCfg.vis_out_dir, f"{stem}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

        pbar.update(1)

    pbar.close()

    if count_samples == 0:
        count_samples = len(dl)

    avg_bev_time = total_bev_time / count_samples
    avg_pred_time = total_pred_time / count_samples

    print(f"\nAverage BEV conversion time per sample: {avg_bev_time * 1000:.2f} ms")
    print(f"Average inference time per sample: {avg_pred_time * 1000:.2f} ms")
    print(
        f"Total average end-to-end per sample: {(avg_bev_time + avg_pred_time) * 1000:.2f} ms  "
        f"(~{1 / (avg_bev_time + avg_pred_time):.2f} FPS)"
    )

    if have_label > 0:
        print(f"[Cos Loss] mean: {cos_sum / have_label:.6f}")
        print(f"[Pixel MAE] mean: {np.mean(pixel_mae_center_list):.3f}°")
        print(f"[Frame Err] center-peak mean: {np.mean(frame_err_list):.3f}°")
        print(
            f"[Pixel Acc] |Δθ|<1°:{prop_lt(pixel_mae_center_list, 1) * 100:.2f}%  "
            f"|Δθ|<2°:{prop_lt(pixel_mae_center_list, 2) * 100:.2f}%  "
            f"|Δθ|<5°:{prop_lt(pixel_mae_center_list, 5) * 100:.2f}%  "
            f"|Δθ|<10°:{prop_lt(pixel_mae_center_list, 10) * 100:.2f}%  "
            f"|Δθ|<20°:{prop_lt(pixel_mae_center_list, 20) * 100:.2f}%  "
            f"|Δθ|<50°:{prop_lt(pixel_mae_center_list, 50) * 100:.2f}%"
        )
        print(
            f"[Frame Acc] |Δθ|<1°:{prop_lt(frame_err_list, 1) * 100:.2f}%  "
            f"|Δθ|<2°:{prop_lt(frame_err_list, 2) * 100:.2f}%  "
            f"|Δθ|<5°:{prop_lt(frame_err_list, 5) * 100:.2f}%  "
            f"|Δθ|<10°:{prop_lt(frame_err_list, 10) * 100:.2f}%  "
            f"|Δθ|<20°:{prop_lt(frame_err_list, 20) * 100:.2f}%  "
            f"|Δθ|<50°:{prop_lt(frame_err_list, 50) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()