#!/usr/bin/env python3
"""
edm_opt_wave_v2_ssi：阶段 1（直接优化初始噪声）→ SSI 多起点跳跃 → 阶段 2（各起点继续优化）

阶段 1：与 edm_opt_wave_v2 完全一致，从光滑速度场反转初始化 z，优化波场 MSE。
SSI：  对阶段 1 的速度场结果施加 K 次 SSI（不同随机种子），得到 K 个初始噪声及重建速度场，
       并与阶段 1 结果对比可视化。
阶段 2：从 K 个 SSI 初始噪声出发，分别继续优化波场 MSE，各起点独立可视化。

生成器参数（num_steps/sigma_min/max/rho/alpha/solver）全程复用 config 的 edm_sampler 块。
SSI forward ODE（反转方向）可单独配置步数与求解器。

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/edm_opt_wave_v2_ssi/demo_edm_opt_wave_v2_ssi.py \\
    --config exp/edm_opt_wave_v2_ssi/config_edm_opt_wave_v2_ssi.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
from src.core.generate import edm_sampler_ode, edm_sampler_ode_latentgrad, forward_edm_sampler
from src.seismic import seismic_master_forward_modeling


# ===========================================================================
# Utilities
# ===========================================================================

def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def v_denormalize_np(v_norm) -> np.ndarray:
    if isinstance(v_norm, torch.Tensor):
        x = v_norm.detach().float().cpu().numpy()
    else:
        x = np.asarray(v_norm, dtype=np.float32)
    return x * 1500.0 + 3000.0


def v_denormalize_tensor(v_norm: torch.Tensor) -> torch.Tensor:
    return v_norm * 1500.0 + 3000.0


def velocity_mae_ssim(
    pred_np: np.ndarray, target_np: np.ndarray,
    device: torch.device, vmin: float, vmax: float,
) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred_np - target_np)))
    def to_t(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a.astype(np.float32)).view(1, 1, a.shape[0], a.shape[1])
        return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0).to(device)
    ssim_val = float(
        pytorch_ssim.ssim(to_t(pred_np), to_t(target_np), window_size=11, size_average=True).item()
    )
    return mae, ssim_val


def load_target_velocity(npy_path: Path, sample_index: int) -> np.ndarray:
    data = np.load(npy_path)
    return data[sample_index, 0].astype(np.float32)


def build_edm_precond(sc: dict, device: torch.device) -> EDMPrecond:
    model = EDMPrecond(
        config=sc,
        img_resolution=sc["img_resolution"],
        padding_resolution=sc["padding_resolution"],
        img_channels=sc["img_channels"],
        label_dim=sc.get("label_dim", 0),
        use_fp16=False,
        sigma_min=sc["sigma_min"],
        sigma_max=sc["sigma_max"],
        sigma_data=sc["sigma_data"],
        model_type=sc.get("model_type", "DhariwalUNet"),
        model_channels=sc["model_channels"],
        channel_mult=sc["channel_mult"],
        channel_mult_emb=sc["channel_mult_emb"],
        num_blocks=sc["num_blocks"],
        attn_resolutions=sc["attn_resolutions"],
        dropout=sc["dropout"],
        label_dropout=sc["label_dropout"],
    ).to(device)
    train_cfg = sc.get("train") or {}
    if train_cfg.get("load_model", True):
        model_path = _resolve_path(
            train_cfg.get("model_path") or "pretrained_model/Curvevel-B_EDM/trained_model.pth"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"EDM model not found: {model_path}")
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    return model


def sample_no_grad(net, z, sigma_max, num_steps, sigma_min, rho, alpha, solver) -> torch.Tensor:
    """z: unit-scale (1,1,H,W) → normalized velocity (H,W), no gradient"""
    latents = z * sigma_max
    out = edm_sampler_ode(
        net=net, latents=latents, class_labels=None,
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
        rho=rho, alpha=alpha, solver=solver,
    )
    while out.dim() > 2:
        out = out.squeeze(0)
    return out


def sample_latent_grad(net, z, sigma_max, num_steps, sigma_min, rho, alpha, solver) -> torch.Tensor:
    """z: unit-scale (1,1,H,W) → normalized velocity (H,W), with latent gradient"""
    latents = z * sigma_max
    out = edm_sampler_ode(
        net=net, latents=latents, class_labels=None,
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
        rho=rho, alpha=alpha, solver=solver,
    )
    while out.dim() > 2:
        out = out.squeeze(0)
    return out


def forward_wave(pred_norm: torch.Tensor) -> torch.Tensor:
    v_phys = v_denormalize_tensor(pred_norm.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


def init_z_from_smooth_velocity(
    net, target_vel_np, smooth_sigma, sigma_mid, sigma_max,
    num_steps, rho, alpha, seed, device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    光滑速度场加噪 + 正向 ODE 初始化 z。
    返回 z_init (unit-scale, shape (1,1,H,W)) 和 v_smooth_phys (m/s)。
    """
    v_smooth_phys = gaussian_filter(target_vel_np, sigma=smooth_sigma)
    v_smooth_norm = (v_smooth_phys - 3000.0) / 1500.0
    v_smooth_t = torch.from_numpy(v_smooth_norm).float().to(device).unsqueeze(0).unsqueeze(0)
    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(v_smooth_t.shape, device=device, dtype=v_smooth_t.dtype, generator=g)
    v_noisy = v_smooth_t + float(sigma_mid) * eps
    with torch.no_grad():
        z_scaled = forward_edm_sampler(
            net, v_noisy, class_labels=None,
            num_steps=num_steps, sigma_min=float(sigma_mid),
            sigma_max=float(sigma_max), rho=float(rho), alpha=float(alpha), solver="heun",
        )
    return z_scaled / float(sigma_max), v_smooth_phys


# ===========================================================================
# Optimization loop
# ===========================================================================

def run_optimization(
    net, z_init: torch.Tensor, target_wave: torch.Tensor, target_vel_np: np.ndarray,
    device: torch.device,
    num_steps: int, sigma_min: float, sigma_max: float, rho: float, alpha: float, solver: str,
    opt_steps: int, lr: float, snapshots: int,
    vel_vmin: float, vel_vmax: float, wave_plot_shot: int,
    out_dir: Path, stage_label: str,
) -> dict:
    """
    通用优化循环（阶段 1 / 阶段 2 均调用）。
    输出：evolution.png、result.png、metrics.png。
    返回 dict: z_final, v_final_np, losses, mae_hist, ssim_hist, grad_hist。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    z = torch.nn.Parameter(z_init.clone().detach().requires_grad_(True))
    z_init_np = z.detach().cpu().numpy().squeeze().copy()

    opt = torch.optim.Adam([z], lr=lr)
    snap_times = np.unique(
        np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int)
    )
    snap_times = np.clip(snap_times, 0, opt_steps)

    hist_vel: list[np.ndarray] = []
    hist_z: list[np.ndarray] = []
    hist_labels: list[str] = []

    def capture(label: str) -> None:
        with torch.no_grad():
            v = sample_no_grad(net, z.detach(), sigma_max, num_steps, sigma_min, rho, alpha, solver)
        hist_vel.append(v_denormalize_np(v).copy())
        hist_z.append(z.detach().cpu().numpy().squeeze().copy())
        hist_labels.append(label)

    losses: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    grad_hist: list[float] = []

    if 0 in snap_times:
        capture("iter 0")

    print(f"\n=== {stage_label}  opt_steps={opt_steps}  lr={lr} ===", flush=True)

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        pred_n = sample_latent_grad(net, z, sigma_max, num_steps, sigma_min, rho, alpha, solver)
        pred_wave = forward_wave(pred_n)
        loss = F.mse_loss(pred_wave, target_wave)
        loss.backward()
        grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
        opt.step()

        losses.append(loss.item())
        grad_hist.append(grad_norm)

        with torch.no_grad():
            pv = v_denormalize_np(pred_n)
            mae_v, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae_v)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            capture(f"iter {it + 1}")

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            print(
                f"  [{stage_label}] iter {it+1}/{opt_steps}  "
                f"wave MSE={loss.item():.6f}  MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  "
                f"||∇z||={grad_norm:.4e}",
                flush=True,
            )

    # --- 图 1：优化演化快照 ---
    ncols = 1 + len(hist_vel)
    z_lim = max(3.0, float(np.abs(z_init_np).max()) + 0.1,
                max((float(np.abs(hz).max()) for hz in hist_z), default=0.0) + 0.1)
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5.5))
    axes = np.atleast_2d(axes)
    # 列 0：True v + z_init
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("True v", fontsize=8)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(z_init_np, cmap="coolwarm", aspect="auto", vmin=-z_lim, vmax=z_lim)
    axes[1, 0].set_title("z init", fontsize=8)
    axes[1, 0].axis("off")
    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("noise z", fontsize=9)
    # 快照列
    for j, (vel, zj, lbl) in enumerate(zip(hist_vel, hist_z, hist_labels)):
        col = j + 1
        axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(lbl, fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-z_lim, vmax=z_lim)
        axes[1, col].axis("off")
    plt.suptitle(f"{stage_label} — optimization evolution", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "evolution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 图 2：最终结果（速度 + 波场）---
    with torch.no_grad():
        pred_n_final = sample_no_grad(net, z.detach(), sigma_max, num_steps, sigma_min, rho, alpha, solver)
        pred_wave_final = forward_wave(pred_n_final)
        pred_vel_np = v_denormalize_np(pred_n_final)
    err_v = pred_vel_np - target_vel_np
    err_lim = float(np.max(np.abs(err_v))) + 1e-6
    sh = wave_plot_shot
    tw = target_wave[sh].detach().cpu().numpy()
    pw = pred_wave_final[sh].detach().cpu().numpy()
    ew = pw - tw
    wlim = max(float(np.abs(tw).max()), float(np.abs(pw).max())) + 1e-6
    elw = float(np.abs(ew).max()) + 1e-6
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np, "target v (m/s)", "viridis", vel_vmin, vel_vmax),
        (pred_vel_np,   "pred v (m/s)",   "viridis", vel_vmin, vel_vmax),
        (err_v,         "v error (m/s)",  "coolwarm", -err_lim, err_lim),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[1], [
        (tw.T, f"target wave shot {sh}", "seismic", -wlim, wlim),
        (pw.T, f"pred wave shot {sh}",   "seismic", -wlim, wlim),
        (ew.T, "wave residual",          "seismic", -elw,  elw),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(
        f"{stage_label} Final | wave MSE={losses[-1]:.6f}  "
        f"vel MAE={mae_hist[-1]:.1f} m/s  SSIM={ssim_hist[-1]:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 图 3：指标曲线 ---
    iters = list(range(1, len(losses) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    axes[0, 0].plot(iters, losses, color="C0")
    axes[0, 0].set_ylabel("wave MSE")
    axes[0, 0].set_title("Wave-space loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(iters, mae_hist, color="C1")
    axes[0, 1].set_ylabel("MAE (m/s)")
    axes[0, 1].set_title("Velocity MAE vs true")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(iters, ssim_hist, color="C2")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_title("Velocity SSIM vs true")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].semilogy(iters, grad_hist, color="C3")
    axes[1, 1].set_ylabel("||∇z|| (log scale)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_title("Gradient norm of z")
    axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle(f"{stage_label} — optimization metrics", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "z_final": z.detach().clone(),
        "v_final_np": pred_vel_np,
        "losses": losses,
        "mae_hist": mae_hist,
        "ssim_hist": ssim_hist,
        "grad_hist": grad_hist,
    }


# ===========================================================================
# SSI
# ===========================================================================

def run_ssi(
    net, v_norm_4d: torch.Tensor,
    sigma_max: float, sigma_min: float, rho: float, alpha: float,
    num_steps: int, solver: str,
    add_noise_sigma: float, forward_num_steps: int, forward_solver: str,
    num_samples: int, seed_base: int, device: torch.device,
) -> list[dict]:
    """
    对归一化速度场 v_norm_4d (1,1,H,W) 施加 K 次 SSI。

    每次 SSI：
      1. 加噪：x_noisy = v_norm_4d + add_noise_sigma * ε
      2. 正向 ODE：forward_edm_sampler(x_noisy, sigma_min=add_noise_sigma → sigma_max)
         → inv_noise（sigma_max 尺度）
      3. 反向 ODE（生成器参数）：edm_sampler_ode(inv_noise) → recon（归一化）
      4. z_ssi = inv_noise / sigma_max（unit-scale，供阶段 2 优化使用）

    返回每项含 k, inv_noise(cpu), z_ssi(cpu), recon_norm(np), recon_phys(np)。
    """
    results = []
    print(
        f"\n=== SSI  K={num_samples}  add_noise_sigma={add_noise_sigma}  "
        f"forward_steps={forward_num_steps} solver={forward_solver}  "
        f"reverse_steps={num_steps} solver={solver} ===",
        flush=True,
    )
    for k in range(num_samples):
        gen = torch.Generator(device=device).manual_seed(seed_base + k)
        eps = torch.randn(v_norm_4d.shape, device=device, dtype=v_norm_4d.dtype, generator=gen)
        x_noisy = v_norm_4d + add_noise_sigma * eps
        with torch.no_grad():
            inv_noise = forward_edm_sampler(
                net, x_noisy, class_labels=None,
                num_steps=forward_num_steps,
                sigma_min=float(add_noise_sigma),
                sigma_max=float(sigma_max),
                rho=float(rho), alpha=float(alpha),
                solver=forward_solver,
            )
            recon = edm_sampler_ode(
                net=net, latents=inv_noise, class_labels=None,
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, alpha=alpha, solver=solver,
            )
        z_ssi = inv_noise / sigma_max
        recon_norm_np = recon.squeeze().detach().cpu().float().numpy()
        recon_phys_np = v_denormalize_np(recon_norm_np)

        print(f"  SSI k={k}  seed={seed_base + k}", flush=True)
        results.append({
            "k": k,
            "seed": seed_base + k,
            "inv_noise": inv_noise.detach().cpu(),
            "z_ssi": z_ssi.detach().cpu(),
            "recon_norm_np": recon_norm_np,
            "recon_phys_np": recon_phys_np,
        })
    return results


def plot_ssi_comparison(
    target_vel_np: np.ndarray,
    stage1_vel_np: np.ndarray,
    stage1_z_np: np.ndarray,
    ssi_results: list[dict],
    device: torch.device,
    vel_vmin: float, vel_vmax: float,
    out_path: Path,
) -> None:
    """
    可视化 SSI 结果：
      行 0（速度场）：True v | 阶段 1 结果 | SSI recon k=0,1,2
      行 1（噪声 z） ：       | 阶段 1 z_final | SSI z_ssi k=0,1,2
      行 2（误差）   ：       | （空）          | |recon_k - stage1_vel| k=0,1,2
    另输出 ssi_cosine.png：K×K 余弦相似矩阵。
    """
    K = len(ssi_results)
    ncols = 2 + K  # True + stage1 + K SSI

    # 噪声 colormap 范围
    z_lim = max(3.0, float(np.abs(stage1_z_np).max()) + 0.1)
    for r in ssi_results:
        z_lim = max(z_lim, float(np.abs(r["z_ssi"].numpy().squeeze()).max()) + 0.1)

    fig, axes = plt.subplots(3, ncols, figsize=(2.8 * ncols, 8.5))
    axes = np.atleast_2d(axes)

    # --- 行 0：速度场 ---
    # 列 0：True v
    im = axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("True v", fontsize=8)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 列 1：阶段 1
    mae_s1, ssim_s1 = velocity_mae_ssim(stage1_vel_np, target_vel_np, device, vel_vmin, vel_vmax)
    im = axes[0, 1].imshow(stage1_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 1].set_title(f"Stage 1\nMAE={mae_s1:.0f} m/s  SSIM={ssim_s1:.3f}", fontsize=8)
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 列 2+：SSI recons
    for i, r in enumerate(ssi_results):
        col = 2 + i
        mae_r, ssim_r = velocity_mae_ssim(r["recon_phys_np"], target_vel_np, device, vel_vmin, vel_vmax)
        mae_vs_s1 = float(np.mean(np.abs(r["recon_phys_np"] - stage1_vel_np)))
        im = axes[0, col].imshow(
            r["recon_phys_np"], cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax
        )
        axes[0, col].set_title(
            f"SSI recon k={i}\nvs true: MAE={mae_r:.0f} m/s\nvs stage1: MAE={mae_vs_s1:.0f} m/s",
            fontsize=7,
        )
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    # --- 行 1：噪声 z ---
    axes[1, 0].axis("off")  # True v 列空
    # 列 1：阶段 1 z_final
    im = axes[1, 1].imshow(stage1_z_np, cmap="coolwarm", aspect="auto", vmin=-z_lim, vmax=z_lim)
    axes[1, 1].set_title("Stage 1 z_final", fontsize=8)
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    # 列 2+：SSI z_ssi
    for i, r in enumerate(ssi_results):
        col = 2 + i
        z_np = r["z_ssi"].numpy().squeeze()
        im = axes[1, col].imshow(z_np, cmap="coolwarm", aspect="auto", vmin=-z_lim, vmax=z_lim)
        axes[1, col].set_title(f"SSI z_ssi k={i}\nseed={r['seed']}", fontsize=8)
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    # --- 行 2：|recon - stage1| 误差图 ---
    axes[2, 0].axis("off")
    axes[2, 1].axis("off")
    for i, r in enumerate(ssi_results):
        col = 2 + i
        err = np.abs(r["recon_phys_np"] - stage1_vel_np)
        elim = float(err.max()) + 1e-6
        im = axes[2, col].imshow(err, cmap="magma", aspect="auto", vmin=0.0, vmax=elim)
        axes[2, col].set_title(f"|SSI recon k={i} − stage1| (m/s)", fontsize=7)
        axes[2, col].axis("off")
        plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("velocity (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("latent z", fontsize=9)
    axes[2, 0].set_ylabel("|recon − stage1|", fontsize=9)
    plt.suptitle("SSI multi-start: Stage 1 result vs SSI reconstructions & starting noises", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SSI comparison saved: {out_path.name}", flush=True)


def plot_ssi_cosine_matrix(ssi_results: list[dict], out_path: Path) -> None:
    """K×K 余弦相似矩阵（基于 inv_noise 展平后的单位向量）"""
    K = len(ssi_results)
    units = []
    for r in ssi_results:
        v = r["inv_noise"].reshape(-1).float()
        units.append(v / (v.norm() + 1e-12))
    V = torch.stack(units, dim=0)
    cos_mat = (V @ V.T).numpy()

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    labels = [f"k={r['k']}\nseed={r['seed']}" for r in ssi_results]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("SSI inv_noise cosine similarity", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stage2_combined_metrics(
    stage1_result: dict,
    stage2_results: list[dict],
    out_path: Path,
) -> None:
    """
    将阶段 1 和各阶段 2 轨迹的指标曲线叠加到同一图中，便于对比。
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def _plot(ax, hist, label, color, linestyle="-"):
        iters = list(range(1, len(hist) + 1))
        ax.plot(iters, hist, color=color, linestyle=linestyle, label=label, lw=1.5)

    colors_s2 = ["C1", "C2", "C3"]
    _plot(axes[0], stage1_result["losses"],    "Stage 1", "C0", "--")
    _plot(axes[1], stage1_result["mae_hist"],  "Stage 1", "C0", "--")
    _plot(axes[2], stage1_result["ssim_hist"], "Stage 1", "C0", "--")

    for i, s2 in enumerate(stage2_results):
        k = s2["k"]
        c = colors_s2[i % len(colors_s2)]
        _plot(axes[0], s2["losses"],    f"Stage 2 k={k}", c)
        _plot(axes[1], s2["mae_hist"],  f"Stage 2 k={k}", c)
        _plot(axes[2], s2["ssim_hist"], f"Stage 2 k={k}", c)

    axes[0].set_title("Wave MSE");  axes[0].set_ylabel("loss"); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Velocity MAE (m/s)"); axes[1].set_ylabel("MAE"); axes[1].grid(True, alpha=0.3)
    axes[2].set_title("Velocity SSIM");  axes[2].set_ylabel("SSIM"); axes[2].grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=7)

    plt.suptitle("Stage 1 vs Stage 2 (SSI multi-start) — optimization curves", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="edm_opt_wave_v2_ssi: 阶段1 z 优化 → SSI 多起点 → 阶段2 各起点继续优化"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_edm_opt_wave_v2_ssi.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path.resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str)
    cp.cuda.Device(device.index if device.index is not None else 0).use()

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 模型
    scfg = cfg["sampler"]
    net = build_edm_precond(scfg, device)
    net.eval()

    # 生成器参数（全程共用）
    es = cfg.get("edm_sampler") or {}
    num_steps = int(es.get("num_steps", 6))
    sigma_min  = float(es.get("sigma_min", 0.002))
    sigma_max  = float(es.get("sigma_max", 80.0))
    rho        = float(es.get("rho", 7.0))
    alpha      = float(es.get("alpha", 1.0))
    solver     = str(es.get("solver", "euler"))

    # z 初始化参数
    zi_cfg       = cfg.get("z_init") or {}
    zi_type      = str(zi_cfg.get("type", "smooth_invert"))
    smooth_sigma = float(zi_cfg.get("smooth_sigma", 10.0))
    sigma_mid    = float(zi_cfg.get("sigma_mid", 1.0))
    zi_steps     = int(zi_cfg.get("num_steps", 100))
    zi_rho       = float(zi_cfg.get("rho", 7.0))
    zi_alpha     = float(zi_cfg.get("alpha", 1.0))
    zi_seed      = int(zi_cfg.get("seed", seed + 7777))

    # 优化参数
    opt_cfg  = cfg.get("optimization") or {}
    s1_cfg   = opt_cfg.get("stage1") or {}
    s2_cfg   = opt_cfg.get("stage2") or {}
    s1_steps = int(s1_cfg.get("opt_steps", 100))
    s1_lr    = float(s1_cfg.get("lr", 1.0))
    s1_snaps = int(s1_cfg.get("snapshots", 8))
    s2_steps = int(s2_cfg.get("opt_steps", 100))
    s2_lr    = float(s2_cfg.get("lr", 1.0))
    s2_snaps = int(s2_cfg.get("snapshots", 8))

    # SSI 参数
    ssi_cfg         = cfg.get("ssi") or {}
    ssi_num         = int(ssi_cfg.get("num_samples", 3))
    add_noise_sigma = float(ssi_cfg.get("add_noise_sigma", 0.5))
    fwd_steps       = int(ssi_cfg.get("forward_num_steps", 100))
    fwd_solver      = str(ssi_cfg.get("forward_solver", "euler"))
    ssi_seed_base   = int(ssi_cfg.get("seed_base", 9000))

    # 可视化参数
    viz         = cfg.get("visualization") or {}
    vel_vmin    = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax    = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_shot   = int(viz.get("wave_plot_shot", 0))

    # 数据
    cv_cfg      = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv_cfg["model60_path"])
    sample_index = int(cv_cfg.get("sample_index", 0))
    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")
    target_vel_np = load_target_velocity(model60_path, sample_index)
    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # 输出目录
    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"edm_opt_wave_v2_ssi_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out_dir / "config.yaml")
    print(f"\nOutput: {out_dir}", flush=True)

    # -----------------------------------------------------------------------
    # 阶段 1：z 初始化 + 优化
    # -----------------------------------------------------------------------
    img_res = int(scfg.get("img_resolution", 70))

    if zi_type == "gaussian":
        print(f"\n=== 阶段 1 z 初始化  type=gaussian  seed={zi_seed} ===", flush=True)
        g = torch.Generator(device=device).manual_seed(zi_seed)
        z_init_tensor = torch.randn(1, 1, img_res, img_res, device=device, generator=g)
    else:
        print(
            f"\n=== 阶段 1 z 初始化  type=smooth_invert  smooth_sigma={smooth_sigma}  "
            f"sigma_mid={sigma_mid}  forward_steps={zi_steps} ===",
            flush=True,
        )
        z_init_tensor, _ = init_z_from_smooth_velocity(
            net, target_vel_np, smooth_sigma, sigma_mid, sigma_max,
            zi_steps, zi_rho, zi_alpha, zi_seed, device,
        )

    with torch.no_grad():
        v_from_zinit = sample_no_grad(net, z_init_tensor, sigma_max, num_steps, sigma_min, rho, alpha, solver)
        v_from_zinit_np = v_denormalize_np(v_from_zinit)
    mae_init, ssim_init = velocity_mae_ssim(v_from_zinit_np, target_vel_np, device, vel_vmin, vel_vmax)
    print(f"  ODE(z_init) vs true: MAE={mae_init:.1f} m/s  SSIM={ssim_init:.4f}", flush=True)

    s1_dir = out_dir / "stage1"
    stage1_result = run_optimization(
        net, z_init_tensor, target_wave, target_vel_np, device,
        num_steps, sigma_min, sigma_max, rho, alpha, solver,
        s1_steps, s1_lr, s1_snaps,
        vel_vmin, vel_vmax, wave_shot,
        s1_dir, "Stage 1",
    )
    print(
        f"\n  Stage 1 final: wave MSE={stage1_result['losses'][-1]:.6f}  "
        f"MAE={stage1_result['mae_hist'][-1]:.1f} m/s  "
        f"SSIM={stage1_result['ssim_hist'][-1]:.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # SSI：对阶段 1 速度场施加 K 次 SSI
    # -----------------------------------------------------------------------
    # 将阶段 1 的最终速度场（normalized）转为 (1,1,H,W) 张量，送入 SSI
    v_stage1_norm = stage1_result["v_final_np"]   # (H,W) m/s
    v_stage1_norm_tensor = (
        torch.from_numpy((v_stage1_norm - 3000.0) / 1500.0).float()
        .to(device).unsqueeze(0).unsqueeze(0)
    )

    ssi_results = run_ssi(
        net, v_stage1_norm_tensor,
        sigma_max, sigma_min, rho, alpha, num_steps, solver,
        add_noise_sigma, fwd_steps, fwd_solver,
        ssi_num, ssi_seed_base, device,
    )

    # 阶段 1 z_final（numpy, unit-scale）用于可视化
    stage1_z_np = stage1_result["z_final"].cpu().numpy().squeeze()

    plot_ssi_comparison(
        target_vel_np, stage1_result["v_final_np"], stage1_z_np,
        ssi_results, device, vel_vmin, vel_vmax,
        out_dir / "ssi_comparison.png",
    )
    plot_ssi_cosine_matrix(ssi_results, out_dir / "ssi_cosine.png")

    # -----------------------------------------------------------------------
    # 阶段 2：从各 SSI 初始噪声继续优化
    # -----------------------------------------------------------------------
    stage2_results = []
    for r in ssi_results:
        k = r["k"]
        z_ssi_init = r["z_ssi"].to(device)
        s2_sub_dir = out_dir / "stage2" / f"k{k:02d}"
        s2_res = run_optimization(
            net, z_ssi_init, target_wave, target_vel_np, device,
            num_steps, sigma_min, sigma_max, rho, alpha, solver,
            s2_steps, s2_lr, s2_snaps,
            vel_vmin, vel_vmax, wave_shot,
            s2_sub_dir, f"Stage 2 k={k}",
        )
        s2_res["k"] = k
        stage2_results.append(s2_res)
        print(
            f"\n  Stage 2 k={k} final: wave MSE={s2_res['losses'][-1]:.6f}  "
            f"MAE={s2_res['mae_hist'][-1]:.1f} m/s  SSIM={s2_res['ssim_hist'][-1]:.4f}",
            flush=True,
        )

    # 汇总对比曲线
    plot_stage2_combined_metrics(
        stage1_result, stage2_results, out_dir / "combined_metrics.png"
    )

    # -----------------------------------------------------------------------
    # 汇总 JSON
    # -----------------------------------------------------------------------
    summary = {
        "stage1": {
            "z_init_type":      zi_type,
            "final_wave_mse":   stage1_result["losses"][-1],
            "final_mae_m_s":    stage1_result["mae_hist"][-1],
            "final_ssim":       stage1_result["ssim_hist"][-1],
            "z_init_mae_m_s":   mae_init,
            "z_init_ssim":      ssim_init,
        },
        "ssi": {
            "num_samples":       ssi_num,
            "add_noise_sigma":   add_noise_sigma,
            "forward_num_steps": fwd_steps,
            "forward_solver":    fwd_solver,
            "seeds": [r["seed"] for r in ssi_results],
        },
        "stage2": [
            {
                "k":              s2["k"],
                "final_wave_mse": s2["losses"][-1],
                "final_mae_m_s":  s2["mae_hist"][-1],
                "final_ssim":     s2["ssim_hist"][-1],
            }
            for s2 in stage2_results
        ],
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done. Output: {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
