"""
残差辅助速度场 v* + 生成潜变量 z 的联合优化。

**核心思路：**
  速度场分解为 v_combined = v0 + v*，其中
    - v0 = EDM(z * σ_max)：由潜变量 z 通过扩散 ODE 生成的速度场
    - v*：残差辅助变量，捕捉生成速度场与目标之间的残差

**初始化：**
  - z0 ~ N(0, 1)
  - v0_init = EDM(z0 * σ_max).detach()          （不带梯度，仅用于初始化）
  - v*_init = gaussian_smooth(v_true, σ) - v0_init （令 v_combined 初始为平滑真值）

**损失函数：**
  L = L_wave(clamp(v0 + v*, -1, 1))                        [第一项]
    + λ_align · MSE( (v0 + v*).detach(), v0 )               [第二项]

  第一项：对 z（经 v0）和 v* 均反传梯度，驱动 v_combined 拟合观测波场。
  第二项：将当前 v_combined 作为固定目标，通过 v0 对 z 反传，驱动 z 逐步"学会"
          生成当前的 v_combined。
          注意：若不对 (v0+v*) 做 detach，则
            d/dz ||(v0+v*) - v0||² = d/dz ||v*||² = 0，z 得不到梯度。

运行（Manifold_constrained_FWI 目录）:
  uv run python exp/edm_residual_joint_opt/demo_edm_residual_joint_opt.py \\
    --config exp/edm_residual_joint_opt/config_edm_residual_joint_opt.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
from src.core.generate import edm_sampler_ode_latentgrad
from src.core.loss import WavefieldLoss
from src.seismic import seismic_master_forward_modeling

try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

def v_denormalize(v_norm: torch.Tensor) -> torch.Tensor:
    return v_norm * 1500.0 + 3000.0


def velocity_mae_ssim(target: torch.Tensor, pred: torch.Tensor):
    t = target.detach().float()
    p = pred.detach().float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    if p.dim() == 2:
        p = p.unsqueeze(0).unsqueeze(0)
    elif p.dim() == 3:
        p = p.unsqueeze(0)
    if p.shape != t.shape:
        p = p.reshape(t.shape)
    mae_norm = torch.mean(torch.abs(t - p)).item()
    ssim_v = pytorch_ssim.ssim(t, p, window_size=11, size_average=True).item()
    mae_phys = torch.mean(torch.abs(v_denormalize(t) - v_denormalize(p))).item()
    return mae_norm, mae_phys, ssim_v


def fmt_metrics(mae_n: float, mae_p: float, ssim: float) -> str:
    return f"MAE_n={mae_n:.4f}, MAE={mae_p:.1f} m/s, SSIM={ssim:.4f}"


def load_dataset_sample(data_path: str, file_index: int, sample_index: int, device):
    data_file = Path(data_path) / f"model{file_index}.npy"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.load(data_file)
    target = data[sample_index]
    if target.ndim == 3:
        target = target[0]
    elif target.ndim == 4:
        target = target[0, 0]
    target_norm = (target.astype(np.float32) - 3000.0) / 1500.0
    return torch.from_numpy(target_norm).float().to(device)


def load_observed_wavefield(data_path: str, file_index: int, sample_index: int, device):
    data_file = Path(data_path) / f"data{file_index}.npy"
    if not data_file.exists():
        raise FileNotFoundError(f"Wavefield data not found: {data_file}")
    data = np.load(data_file)
    return torch.from_numpy(data[sample_index].astype(np.float32)).float().to(device)


def forward_velocity_to_wavefield(v_norm: torch.Tensor, img_res: int) -> torch.Tensor:
    velocity_physical = v_denormalize(v_norm).clamp(1500.0, 4500.0)
    velocity_2d = velocity_physical.squeeze().reshape(img_res, img_res).to(v_norm.device)
    return seismic_master_forward_modeling(velocity_2d)


def build_wavefield_loss(loss_type: str, w2_cfg: dict) -> WavefieldLoss:
    lt = str(loss_type).lower().strip()
    dt = 0.001
    if lt in ("wavefield_l1", "l1"):
        return WavefieldLoss(loss_type="l1", dt=dt)
    if lt in ("wavefield_mse", "mse"):
        return WavefieldLoss(loss_type="mse", dt=dt)
    if lt in ("wavefield_l2_sq", "l2_sq"):
        return WavefieldLoss(loss_type="l2_sq", dt=dt)
    if lt == "w2_per_trace":
        return WavefieldLoss(
            loss_type="w2_per_trace",
            dt=dt,
            normalize_type=w2_cfg.get("normalize_type", "softplus"),
            b=float(w2_cfg.get("b", 0.1)),
        )
    raise ValueError(f"Unknown loss_type {loss_type!r}")


FWI_LOSS_TYPES = ("wavefield_l1", "l1", "wavefield_mse", "mse", "wavefield_l2_sq", "l2_sq", "w2_per_trace")


def _vel_to_rgb_uint8(arr: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    t = (np.clip(arr.astype(np.float64), vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    rgba = plt.cm.viridis(t)
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def load_edm(model_path: str, sampler_config: dict, device: torch.device) -> EDMPrecond:
    if not os.path.isabs(model_path):
        model_path = str(_MANIFOLD_ROOT / model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    sc = dict(sampler_config) if sampler_config else {}
    model = EDMPrecond(
        config=sc,
        img_resolution=sc.get("img_resolution", 70),
        padding_resolution=sc.get("padding_resolution", 72),
        img_channels=sc.get("img_channels", 1),
        label_dim=sc.get("label_dim", 0),
        use_fp16=False,
        sigma_min=sc.get("sigma_min", 0.002),
        sigma_max=sc.get("sigma_max", 80.0),
        sigma_data=sc.get("sigma_data", 0.6),
        model_type=sc.get("model_type", "DhariwalUNet"),
        model_channels=sc.get("model_channels", 32),
        channel_mult=sc.get("channel_mult", [1, 2, 3, 4]),
        channel_mult_emb=sc.get("channel_mult_emb", 4),
        num_blocks=sc.get("num_blocks", 3),
        attn_resolutions=sc.get("attn_resolutions", [32, 16, 8]),
        dropout=sc.get("dropout", 0.0),
        label_dropout=sc.get("label_dropout", 0),
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def sample_ode_latent_grad(
    net: EDMPrecond,
    z: torch.Tensor,
    sigma_max: float,
    num_steps: int,
    sigma_min: float,
    rho: float,
    alpha: float,
    solver: str,
) -> torch.Tensor:
    latents = z * sigma_max
    out = edm_sampler_ode_latentgrad(
        net=net,
        latents=latents,
        class_labels=None,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        alpha=alpha,
        solver=solver,
    )
    while out.dim() > 2:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# 核心优化函数
# ---------------------------------------------------------------------------

def run_residual_joint_opt(
    v_star_init: torch.Tensor,
    z_init: torch.Tensor,
    wavefield_loss_fn: WavefieldLoss,
    obs_wf_base: torch.Tensor,
    img_res: int,
    edm: EDMPrecond,
    joint_cfg: dict,
    target_batch: torch.Tensor,
    device: torch.device,
    edm_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    alpha: float,
    solver: str,
) -> dict[str, Any]:
    """
    联合优化 z 和 v*，有效速度场为 v_combined = EDM(z) + v*。

    损失：
      L_wave  = wavefield_loss( clamp(v0 + v*, -1, 1) )
                  → 对 z（经由 v0）和 v* 均反传梯度
      L_align = MSE( (v0 + v*).detach(), v0 )
                  → 以 v_combined 为固定目标，驱动 z 学习生成 v_combined；
                    若不 detach，则 d||v*||²/dz = 0，z 无法更新。
      L_total = L_wave + λ_align * L_align

    返回 dict：
      v_star, z, v0, v_combined    — 最终张量
      loss, data_loss, align_loss  — 损失历史
      mae_combined, ssim_combined  — v_combined vs 真值的指标历史
      mae_v0, ssim_v0              — v0 (EDM 生成) vs 真值的指标历史
      vstar_mag                    — ||v*||_mean 历史（残差大小）
      grad_vstar, grad_z           — 梯度范数历史
      snap_v_comb, snap_v0, snap_v_star, snap_labels — 演化快照
    """
    iters = max(1, int(joint_cfg.get("iterations", 300)))
    lr_vstar = float(joint_cfg.get("lr_vstar", 0.03))
    lr_z = float(joint_cfg.get("lr_z", 0.01))
    align_lambda = float(joint_cfg.get("align_lambda", 1.0))
    loss_scale = float(joint_cfg.get("loss_scale", 1.0))
    log_every = max(1, int(joint_cfg.get("log_every", 10)))
    use_cosine = bool(joint_cfg.get("cosine_annealing", True))
    eta_min = float(joint_cfg.get("eta_min", 0.0))
    snap_every = max(1, int(joint_cfg.get("snap_every", 30)))
    gif_enabled = bool(joint_cfg.get("gif_enabled", True))

    v_star = torch.nn.Parameter(v_star_init.clone().detach().requires_grad_(True))
    z = torch.nn.Parameter(z_init.clone().detach().requires_grad_(True))

    opt = torch.optim.Adam([
        {"params": [v_star], "lr": lr_vstar},
        {"params": [z], "lr": lr_z},
    ])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=eta_min)
        if use_cosine
        else None
    )

    # 追踪历史
    loss_hist: list[float] = []
    data_hist: list[float] = []
    align_hist: list[float] = []
    mae_comb_hist: list[float] = []
    ssim_comb_hist: list[float] = []
    mae_v0_hist: list[float] = []
    ssim_v0_hist: list[float] = []
    vstar_mag_hist: list[float] = []
    grad_vstar_hist: list[float] = []
    grad_z_hist: list[float] = []

    snap_v_comb: list[torch.Tensor] = []
    snap_v0: list[torch.Tensor] = []
    snap_v_star: list[torch.Tensor] = []
    snap_labels: list[str] = []

    v0_final: torch.Tensor | None = None

    # 初始帧（iter 0）
    with torch.no_grad():
        v0_init_sample = sample_ode_latent_grad(
            edm, z, sigma_max, edm_steps, sigma_min, rho, alpha, solver
        ).view(1, 1, img_res, img_res)
        v_comb_init = (v0_init_sample + v_star).clamp(-1.0, 1.0)
    snap_v_comb.append(v_comb_init.cpu().clone())
    snap_v0.append(v0_init_sample.cpu().clone())
    snap_v_star.append(v_star.detach().cpu().clone())
    snap_labels.append("iter 0 (init)")

    for it in range(iters):
        opt.zero_grad(set_to_none=True)

        # v0 = EDM(z)：梯度可流向 z
        v0 = sample_ode_latent_grad(
            edm, z, sigma_max, edm_steps, sigma_min, rho, alpha, solver
        ).view(1, 1, img_res, img_res)

        v_combined = v0 + v_star  # 梯度同时流向 z（经 v0）和 v_star

        # 第一项：波场数据拟合损失
        sim = forward_velocity_to_wavefield(v_combined.clamp(-1.0, 1.0), img_res)
        L_wave = wavefield_loss_fn(sim, obs_wf_base) * loss_scale

        # 第二项：驱动 z 学习当前 v_combined
        # 关键：对 (v0 + v*) 做 detach，作为固定目标
        # 若不 detach：d/dz[(v0+v*) - v0]² = d/dz[v*²] = 0，z 梯度为零
        v_combined_target = v_combined.detach()
        L_align = F.mse_loss(v_combined_target, v0)

        L_total = L_wave + align_lambda * L_align
        L_total.backward()

        # 在 step 前记录梯度范数
        gvs = v_star.grad.norm().item() if v_star.grad is not None else 0.0
        gz = z.grad.norm().item() if z.grad is not None else 0.0
        grad_vstar_hist.append(gvs)
        grad_z_hist.append(gz)

        opt.step()
        if scheduler is not None:
            scheduler.step()

        loss_hist.append(float(L_total.detach().item()))
        data_hist.append(float(L_wave.detach().item()))
        align_hist.append(float(L_align.detach().item()))

        with torch.no_grad():
            v0_d = v0.detach()
            v_comb_d = (v0_d + v_star.detach()).clamp(-1.0, 1.0)
            mae_comb, mae_comb_p, ssim_comb = velocity_mae_ssim(target_batch, v_comb_d)
            mae_v0, mae_v0_p, ssim_v0 = velocity_mae_ssim(target_batch, v0_d)
            vstar_mag = v_star.detach().abs().mean().item()

        mae_comb_hist.append(mae_comb)
        ssim_comb_hist.append(ssim_comb)
        mae_v0_hist.append(mae_v0)
        ssim_v0_hist.append(ssim_v0)
        vstar_mag_hist.append(vstar_mag)
        v0_final = v0_d

        if it % log_every == 0 or it == iters - 1:
            print(
                f"  [residual] {it + 1}/{iters}  total={loss_hist[-1]:.6f}  "
                f"data={data_hist[-1]:.6f}  align={align_hist[-1]:.6f}"
                f"\n    v_combined: {fmt_metrics(mae_comb, mae_comb_p, ssim_comb)}"
                f"  grad_v*={gvs:.4e}  grad_z={gz:.4e}"
                f"\n    v0 (EDM):   {fmt_metrics(mae_v0, mae_v0_p, ssim_v0)}"
                f"  |v*|_mean={vstar_mag:.6f}",
                flush=True,
            )

        # 演化快照
        if gif_enabled and ((it + 1) % snap_every == 0 or it == iters - 1):
            with torch.no_grad():
                snap_v_comb.append((v0_d + v_star.detach()).cpu().clone())
                snap_v0.append(v0_d.view(1, 1, img_res, img_res).cpu().clone())
                snap_v_star.append(v_star.detach().cpu().clone())
            snap_labels.append(f"iter {it + 1}")

    v_star_final = v_star.detach()
    v_combined_final = (
        (v0_final + v_star_final).clamp(-1.0, 1.0)
        if v0_final is not None
        else None
    )

    return dict(
        v_star=v_star_final,
        z=z.detach(),
        v0=v0_final,
        v_combined=v_combined_final,
        loss=loss_hist,
        data_loss=data_hist,
        align_loss=align_hist,
        mae_combined=mae_comb_hist,
        ssim_combined=ssim_comb_hist,
        mae_v0=mae_v0_hist,
        ssim_v0=ssim_v0_hist,
        vstar_mag=vstar_mag_hist,
        grad_vstar=grad_vstar_hist,
        grad_z=grad_z_hist,
        snap_v_comb=snap_v_comb,
        snap_v0=snap_v0,
        snap_v_star=snap_v_star,
        snap_labels=snap_labels,
        gif_enabled=gif_enabled,
    )


# ---------------------------------------------------------------------------
# 可视化函数
# ---------------------------------------------------------------------------

def _imshow(ax, data: np.ndarray, title: str, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_evolution(r: dict, exp_dir: Path) -> None:
    """
    演化可视化（3 行 × N 列快照）：
      行 0: v_combined = v0 + v*
      行 1: v0 (EDM 生成)
      行 2: v* (残差)
    """
    snap_v_comb = r["snap_v_comb"]
    snap_v0 = r["snap_v0"]
    snap_v_star = r["snap_v_star"]
    snap_labels = r["snap_labels"]
    n = len(snap_v_comb)
    if n == 0:
        return

    vm = (-1.0, 1.0)
    fig, axes = plt.subplots(3, n, figsize=(max(2.4 * n, 6), 8.4))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    row_labels = ["v_combined", "v0 (EDM)", "v* (residual)"]
    for j, label in enumerate(snap_labels):
        axes[0, j].imshow(snap_v_comb[j].squeeze().numpy(), cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[0, j].set_title(label, fontsize=7)
        axes[0, j].axis("off")

        axes[1, j].imshow(snap_v0[j].squeeze().numpy(), cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[1, j].axis("off")

        vstar_arr = snap_v_star[j].squeeze().numpy()
        vabs = max(np.abs(vstar_arr).max(), 1e-6)
        axes[2, j].imshow(vstar_arr, cmap="coolwarm", aspect="auto", vmin=-vabs, vmax=vabs)
        axes[2, j].axis("off")

    for row, rl in enumerate(row_labels):
        axes[row, 0].set_ylabel(rl, fontsize=8)

    plt.suptitle("EDM Residual Joint Opt — optimization evolution (normalized μ)", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "evolution.png", dpi=160, bbox_inches="tight")
    plt.close()

    if not r.get("gif_enabled", True):
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio  # type: ignore
        except ImportError:
            print("  [warn] imageio not found, GIF skipped.", flush=True)
            return

    rgb_comb = [_vel_to_rgb_uint8(f.squeeze().numpy()) for f in snap_v_comb]
    imageio.mimsave(exp_dir / "v_combined_evolution.gif", rgb_comb, duration=0.15, loop=0)
    rgb_v0 = [_vel_to_rgb_uint8(f.squeeze().numpy()) for f in snap_v0]
    imageio.mimsave(exp_dir / "v0_evolution.gif", rgb_v0, duration=0.15, loop=0)
    print(f"  GIF ({len(rgb_comb)} frames) saved.", flush=True)


def save_comparison(r: dict, true_np: np.ndarray, init_np: np.ndarray, exp_dir: Path, smooth_sigma: float) -> None:
    """
    对比图（2 行 × 6 列）：
      行 0: True | Init(=v_combined_init) | v_combined | v0(EDM) | v*(残差) | z
      行 1: 空白 | 误差×3               | v* 值域图   | |z| 图
    """
    v_comb_np = r["v_combined"][0, 0].cpu().numpy() if r["v_combined"] is not None else np.zeros_like(true_np)
    v0_np = r["v0"][0, 0].cpu().numpy() if r["v0"] is not None else np.zeros_like(true_np)
    vstar_np = r["v_star"][0, 0].cpu().numpy()
    z_np = r["z"][0, 0].cpu().numpy()

    mn_c = r["final_metrics_combined"]
    mn_v0 = r["final_metrics_v0"]

    vm = (-1.0, 1.0)
    fig, axes = plt.subplots(2, 6, figsize=(17, 5.5))

    # 行 0
    _imshow(axes[0, 0], true_np, "True μ", vmin=vm[0], vmax=vm[1])
    _imshow(axes[0, 1], init_np, f"Init (v_comb, σ={smooth_sigma:g})", vmin=vm[0], vmax=vm[1])
    _imshow(axes[0, 2], v_comb_np,
            f"v_combined\nMAE_n={mn_c[0]:.4f}  SSIM={mn_c[2]:.4f}", vmin=vm[0], vmax=vm[1])
    _imshow(axes[0, 3], v0_np,
            f"v0 (EDM)\nMAE_n={mn_v0[0]:.4f}  SSIM={mn_v0[2]:.4f}", vmin=vm[0], vmax=vm[1])
    vabs = max(np.abs(vstar_np).max(), 1e-6)
    _imshow(axes[0, 4], vstar_np, f"v* (residual)\n|v*|_mean={np.abs(vstar_np).mean():.4f}",
            cmap="coolwarm", vmin=-vabs, vmax=vabs)
    _imshow(axes[0, 5], z_np, f"z (latent)\n|z|_max={np.abs(z_np).max():.2f}", cmap="RdBu_r")

    # 行 1：误差图
    axes[1, 0].axis("off")
    for col, (pred, label) in enumerate(
        [(init_np, "Init error"), (v_comb_np, "v_combined err"), (v0_np, "v0 err")],
        start=1,
    ):
        err = pred - true_np
        el = max(np.abs(err).max(), 1e-6)
        axes[1, col].imshow(err, cmap="coolwarm", aspect="auto", vmin=-el, vmax=el)
        axes[1, col].set_title(label, fontsize=7)
        axes[1, col].axis("off")
    _imshow(axes[1, 4], np.abs(vstar_np), "|v*|", cmap="hot")
    _imshow(axes[1, 5], np.abs(z_np), "|z|", cmap="hot")

    plt.suptitle("EDM Residual Joint Opt — velocity comparison (normalized μ)", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_trajectories(r: dict, exp_dir: Path) -> None:
    """
    轨迹图（2×3 六宫格）：
      (0,0) MAE — v_combined vs v0
      (0,1) SSIM — v_combined vs v0
      (0,2) ||v*||_mean — 残差大小变化
      (1,0) Loss — total / data / align
      (1,1) Grad norm — ||∇v*|| / ||∇z||
      (1,2) align_loss 单独（便于观察收敛）
    """
    iters = list(range(1, len(r["mae_combined"]) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    # (0,0) MAE
    ax = axes[0, 0]
    ax.plot(iters, r["mae_combined"], color="steelblue", label="v_combined")
    ax.plot(iters, r["mae_v0"], color="darkorange", linestyle="--", label="v0 (EDM)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("MAE vs True")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) SSIM
    ax = axes[0, 1]
    ax.plot(iters, r["ssim_combined"], color="steelblue", label="v_combined")
    ax.plot(iters, r["ssim_v0"], color="darkorange", linestyle="--", label="v0 (EDM)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs True")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) 残差大小
    ax = axes[0, 2]
    ax.plot(iters, r["vstar_mag"], color="purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|v*| mean")
    ax.set_title("Residual magnitude ||v*||_mean")
    ax.grid(True, alpha=0.3)

    # (1,0) 损失
    ax = axes[1, 0]
    ax.plot(iters, r["loss"], color="black", label="total")
    ax.plot(iters, r["data_loss"], color="steelblue", linestyle="--", label="data")
    ax.plot(iters, r["align_loss_scaled"], color="darkorange", linestyle=":", label="align (scaled)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) 梯度范数
    ax = axes[1, 1]
    ax.semilogy(iters, r["grad_vstar"], color="steelblue", label="||∇v*||")
    if any(g > 0 for g in r["grad_z"]):
        ax.semilogy(iters, r["grad_z"], color="darkorange", linestyle="--", label="||∇z||")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm (log scale)")
    ax.set_title("Gradient norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) align_loss 原始值
    ax = axes[1, 2]
    ax.plot(iters, r["align_loss"], color="darkorange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE(v_combined.detach(), v0)")
    ax.set_title("Alignment loss (raw)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("EDM Residual Joint Opt — optimization trajectories", fontsize=10)
    plt.tight_layout()
    plt.savefig(exp_dir / "trajectories.png", dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is required for wavefield forward modeling.")

    parser = argparse.ArgumentParser(description="EDM residual auxiliary variable joint optimization")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_SCRIPT_DIR / "config_edm_residual_joint_opt.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        cp.cuda.Device(device.index if device.index is not None else 0).use()

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 数据路径
    data_path = cfg.get("target", {}).get("data_path") or str(_MANIFOLD_ROOT / "data" / "Curvevel-B")
    file_index = int(cfg.get("target", {}).get("file_index", 60))
    sample_index = int(cfg.get("target", {}).get("sample_index", 5))

    sampler_config = cfg.get("sampler", {})
    img_res = int(sampler_config.get("img_resolution", 70))

    init_cfg = cfg.get("init") or {}
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))

    fwi_common = cfg.get("fwi_common") or {}
    fwi_loss_type = str(fwi_common.get("loss_type", "mse")).lower().strip()
    if fwi_loss_type not in FWI_LOSS_TYPES:
        raise ValueError(f"fwi_common.loss_type must be one of {FWI_LOSS_TYPES}")

    w2_global = cfg.get("w2_per_trace") or {}
    wavefield_loss_fn = build_wavefield_loss(fwi_loss_type, w2_global)

    # 加载数据
    observed = load_observed_wavefield(data_path, file_index, sample_index, device)
    target = load_dataset_sample(data_path, file_index, sample_index, device)
    target_batch = target.unsqueeze(0).unsqueeze(0)  # (1, 1, 70, 70)

    from scipy.ndimage import gaussian_filter

    t_np = target_batch[0, 0].detach().cpu().numpy()
    v_smooth = (
        torch.from_numpy(gaussian_filter(t_np, sigma=smooth_sigma))
        .float()
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, 70, 70)

    # 加载 EDM 模型
    model_path = cfg["model"]["model_path"]
    if not os.path.isabs(model_path):
        model_path = str(_MANIFOLD_ROOT / model_path)
    edm = load_edm(model_path, sampler_config, device)

    es = cfg.get("edm_sampler") or {}
    edm_num_steps = int(es.get("num_steps", 4))
    edm_sigma_min = float(es.get("sigma_min", 0.002))
    edm_sigma_max = float(es.get("sigma_max", 80.0))
    edm_rho = float(es.get("rho", 7.0))
    edm_alpha = float(es.get("alpha", 1.0))
    edm_solver = str(es.get("solver", "euler")).lower()

    # 初始化 z0
    joint_z_cfg = cfg.get("joint_z") or {}
    z_seed = int(joint_z_cfg.get("seed", seed + 9000))
    g_z = torch.Generator(device=device).manual_seed(z_seed)
    z_init = torch.randn(1, 1, img_res, img_res, device=device, dtype=torch.float32, generator=g_z)

    # 初始化 v0 = EDM(z0)，然后 v* = v_smooth - v0
    print("Initializing v0 = EDM(z0) ...", flush=True)
    with torch.no_grad():
        v0_init = sample_ode_latent_grad(
            edm, z_init, edm_sigma_max, edm_num_steps, edm_sigma_min, edm_rho, edm_alpha, edm_solver
        ).view(1, 1, img_res, img_res)
    v_star_init = v_smooth - v0_init  # v_combined 初始 = v_smooth

    init_mae_n, init_mae_p, init_ssim = velocity_mae_ssim(target_batch, v_smooth)
    v0_init_mae_n, v0_init_mae_p, v0_init_ssim = velocity_mae_ssim(target_batch, v0_init)
    print(
        f"Init v_combined (= v_smooth): {fmt_metrics(init_mae_n, init_mae_p, init_ssim)}"
        f"\nInit v0 (EDM):               {fmt_metrics(v0_init_mae_n, v0_init_mae_p, v0_init_ssim)}"
        f"\nInit |v*|_mean = {v_star_init.abs().mean().item():.6f}",
        flush=True,
    )

    # 读取实验配置
    ex_block = cfg.get("experiments") or {}
    if "residual_joint" not in ex_block:
        raise ValueError("config.experiments.residual_joint is required")
    mcfg = {**fwi_common, **ex_block["residual_joint"]}
    align_lambda = float(mcfg.get("align_lambda", 1.0))

    print(
        f"\n=== edm_residual_joint_opt  |  file={file_index} sample={sample_index}  "
        f"smooth_σ={smooth_sigma}  λ_align={align_lambda} ===\n",
        flush=True,
    )

    # 运行优化
    res = run_residual_joint_opt(
        v_star_init=v_star_init,
        z_init=z_init,
        wavefield_loss_fn=wavefield_loss_fn,
        obs_wf_base=observed,
        img_res=img_res,
        edm=edm,
        joint_cfg=mcfg,
        target_batch=target_batch,
        device=device,
        edm_steps=edm_num_steps,
        sigma_min=edm_sigma_min,
        sigma_max=edm_sigma_max,
        rho=edm_rho,
        alpha=edm_alpha,
        solver=edm_solver,
    )

    # 最终指标
    final_metrics_combined = velocity_mae_ssim(target_batch, res["v_combined"])
    final_metrics_v0 = velocity_mae_ssim(target_batch, res["v0"]) if res["v0"] is not None else (0.0, 0.0, 0.0)
    res["final_metrics_combined"] = final_metrics_combined
    res["final_metrics_v0"] = final_metrics_v0

    mn_c = final_metrics_combined
    mn_v0 = final_metrics_v0
    print(
        f"\nFinal v_combined: {fmt_metrics(mn_c[0], mn_c[1], mn_c[2])}"
        f"\nFinal v0 (EDM):   {fmt_metrics(mn_v0[0], mn_v0[1], mn_v0[2])}"
        f"\nFinal |v*|_mean = {res['v_star'].abs().mean().item():.6f}",
        flush=True,
    )

    # 为轨迹图准备缩放后的 align_loss
    res["align_loss_scaled"] = [v * align_lambda for v in res["align_loss"]]

    # -----------------------------------------------------------------------
    # 保存输出
    # -----------------------------------------------------------------------
    outdir = cfg.get("paths", {}).get("outdir", "demo_output")
    if not os.path.isabs(outdir):
        outdir = str(_SCRIPT_DIR / outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(outdir) / f"residual_f{file_index}_s{sample_index}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, exp_dir / "config.yaml")

    true_np = t_np
    init_np = v_smooth[0, 0].cpu().numpy()

    save_comparison(res, true_np, init_np, exp_dir, smooth_sigma)
    save_trajectories(res, exp_dir)
    save_evolution(res, exp_dir)

    # metrics.csv
    with open(exp_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["var", "mae_norm", "mae_phys_m_s", "ssim"])
        w.writerow(["v_combined", *final_metrics_combined])
        w.writerow(["v0_edm", *final_metrics_v0])
        w.writerow(["init_v_smooth", init_mae_n, init_mae_p, init_ssim])

    # summary.json
    summary: dict[str, Any] = {
        "exp_dir": str(exp_dir),
        "file_index": file_index,
        "sample_index": sample_index,
        "align_lambda": align_lambda,
        "metrics": {
            "v_combined": list(final_metrics_combined),
            "v0_edm": list(final_metrics_v0),
            "init_v_smooth": [init_mae_n, init_mae_p, init_ssim],
        },
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
