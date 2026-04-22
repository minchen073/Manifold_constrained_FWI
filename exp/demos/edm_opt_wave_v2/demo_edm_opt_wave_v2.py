#!/usr/bin/env python3
"""
edm_opt_wave v2：初始噪声 z 由高斯光滑速度场加噪反转得到，而非纯随机高斯。

初始化流程：
  1. target_vel_smooth = GaussianFilter(target_vel, sigma=smooth_sigma)
  2. v_noisy = v_smooth_norm + sigma_mid * eps            （中间时刻跳跃）
  3. z_scaled = forward_edm_sampler(v_noisy,              （正向 ODE: sigma_mid → sigma_max）
                    sigma_min=sigma_mid, sigma_max=sigma_max)
  4. z_init   = z_scaled / sigma_max                     （单位尺度，与优化变量同尺度）

其余优化逻辑与 edm_opt_wave 完全一致。

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/edm_opt_wave_v2/demo_edm_opt_wave_v2.py \\
    --config exp/edm_opt_wave_v2/config_edm_opt_wave_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import math
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


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def v_denormalize_tensor(v_norm: torch.Tensor) -> torch.Tensor:
    return v_norm * 1500.0 + 3000.0


def v_denormalize_np(v_norm: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(v_norm, torch.Tensor):
        x = v_norm.detach().float().cpu().numpy()
    else:
        x = v_norm.astype(np.float32)
    return x * 1500.0 + 3000.0


def vel_to_ssim_tensor(vel_np: np.ndarray, vmin: float, vmax: float) -> torch.Tensor:
    t = torch.from_numpy(vel_np.astype(np.float32)).view(1, 1, vel_np.shape[0], vel_np.shape[1])
    return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0)


def velocity_mae_ssim(
    pred_vel_np: np.ndarray,
    target_vel_np: np.ndarray,
    device: torch.device,
    vmin: float,
    vmax: float,
) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred_vel_np - target_vel_np)))
    t1 = vel_to_ssim_tensor(pred_vel_np, vmin, vmax).to(device)
    t2 = vel_to_ssim_tensor(target_vel_np, vmin, vmax).to(device)
    ssim_val = float(pytorch_ssim.ssim(t1, t2, window_size=11, size_average=True).item())
    return mae, ssim_val


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
    latents = z * sigma_max
    out = edm_sampler_ode_latentgrad(
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


def load_target_velocity(npy_path: Path, sample_index: int) -> np.ndarray:
    data = np.load(npy_path)  # (N, 1, 70, 70)
    return data[sample_index, 0].astype(np.float32)


# ---------------------------------------------------------------------------
# z 初始化：光滑速度场 → 中间时刻跳跃 → 正向 ODE → z_init（单位尺度）
# ---------------------------------------------------------------------------

def init_z_from_smooth_velocity(
    net: EDMPrecond,
    target_vel_np: np.ndarray,
    smooth_sigma: float,
    sigma_mid: float,
    sigma_max: float,
    num_steps: int,
    rho: float,
    alpha: float,
    seed: int,
    device: torch.device,
    img_res: int = 70,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    从目标速度场的高斯光滑版本出发，通过中间时刻跳跃 + 正向 ODE 初始化 z。

    返回：
      z_init  — 单位尺度张量 (1, 1, img_res, img_res)，用作优化变量初值
      v_smooth_np — 光滑速度场 (m/s)，用于可视化
    """
    # 光滑化（物理空间）→ 归一化
    v_smooth_phys = gaussian_filter(target_vel_np, sigma=smooth_sigma)
    v_smooth_norm = (v_smooth_phys - 3000.0) / 1500.0
    v_smooth_t = (
        torch.from_numpy(v_smooth_norm).float().to(device)
        .unsqueeze(0).unsqueeze(0)
    )  # (1, 1, H, W)

    # VE 噪声跳跃到 sigma_mid 时刻
    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(v_smooth_t.shape, device=device, dtype=v_smooth_t.dtype, generator=g)
    v_noisy = v_smooth_t + float(sigma_mid) * eps

    # 正向 ODE：sigma_mid → sigma_max
    with torch.no_grad():
        z_scaled = forward_edm_sampler(
            net, v_noisy,
            class_labels=None,
            num_steps=num_steps,
            sigma_min=float(sigma_mid),
            sigma_max=float(sigma_max),
            rho=float(rho),
            alpha=float(alpha),
            solver="heun",
        )

    z_init = z_scaled / float(sigma_max)  # 单位尺度
    return z_init, v_smooth_phys


def main() -> None:
    parser = argparse.ArgumentParser(description="EDM 初始噪声优化（波场 MSE，光滑速度场反转初始化）")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_edm_opt_wave_v2.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path.resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (CuPy + wave forward).")

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str)
    cp.cuda.Device(device.index if device.index is not None else 0).use()

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    scfg = cfg["sampler"]
    sampler = build_edm_precond(scfg, device)
    sampler.eval()

    es = cfg.get("edm_sampler") or {}
    num_steps = int(es.get("num_steps", 20))
    sigma_min = float(es.get("sigma_min", 0.002))
    sigma_max = float(es.get("sigma_max", 80.0))
    rho = float(es.get("rho", 7.0))
    alpha = float(es.get("alpha", 1.0))
    solver = str(es.get("solver", "heun"))
    img_res = int(scfg.get("img_resolution", 70))

    cv = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    sample_index = int(cv.get("sample_index", 0))

    opt_cfg = cfg.get("optimization") or {}
    opt_steps = int(opt_cfg.get("opt_steps", 100))
    lr = float(opt_cfg.get("lr", 0.05))
    snapshots = int(opt_cfg.get("snapshots", 10))
    reg_weight = float(opt_cfg.get("reg_weight", 0.0))

    z_init_cfg = cfg.get("z_init") or {}
    smooth_sigma = float(z_init_cfg.get("smooth_sigma", 10.0))
    sigma_mid = float(z_init_cfg.get("sigma_mid", 5.0))
    z_init_steps = int(z_init_cfg.get("num_steps", 20))
    z_init_rho = float(z_init_cfg.get("rho", 7.0))
    z_init_alpha = float(z_init_cfg.get("alpha", 1.0))
    z_init_seed = int(z_init_cfg.get("seed", seed + 7777))

    viz = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"edm_opt_wave_v2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)

    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    target_vel_np = load_target_velocity(model60_path, sample_index)

    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # === z 初始化：光滑速度场加噪反转 ===
    print(
        f"\n=== z init: smooth_sigma={smooth_sigma}  sigma_mid={sigma_mid}  "
        f"forward_ode_steps={z_init_steps} ===",
        flush=True,
    )
    z_init_tensor, v_smooth_phys = init_z_from_smooth_velocity(
        sampler, target_vel_np, smooth_sigma, sigma_mid, sigma_max,
        z_init_steps, z_init_rho, z_init_alpha, z_init_seed, device, img_res,
    )

    # 验证初始化质量
    with torch.no_grad():
        v_from_zinit = sample_no_grad(
            sampler, z_init_tensor, sigma_max, num_steps, sigma_min, rho, alpha, solver
        )
        v_from_zinit_np = v_denormalize_np(v_from_zinit)
    mae_init, ssim_init = velocity_mae_ssim(v_from_zinit_np, target_vel_np, device, vel_vmin, vel_vmax)
    mae_smooth, ssim_smooth = velocity_mae_ssim(v_smooth_phys, target_vel_np, device, vel_vmin, vel_vmax)
    print(f"  v_smooth  vs true: MAE={mae_smooth:.1f} m/s  SSIM={ssim_smooth:.4f}", flush=True)
    print(f"  ODE(z_init) vs true: MAE={mae_init:.1f} m/s  SSIM={ssim_init:.4f}", flush=True)

    z = torch.nn.Parameter(z_init_tensor.clone().detach().requires_grad_(True))
    z_init_np = z.detach().cpu().numpy().squeeze().copy()

    opt = torch.optim.Adam([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=0.0)
    snap_times = np.unique(np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int))
    snap_times = np.clip(snap_times, 0, opt_steps)

    hist_vel: list[np.ndarray] = []
    hist_z: list[np.ndarray] = []
    hist_labels: list[str] = []

    def capture(label: str) -> None:
        with torch.no_grad():
            pred_n = sample_no_grad(sampler, z.detach(), sigma_max, num_steps, sigma_min, rho, alpha, solver)
            pred_vel_np = v_denormalize_np(pred_n)
        hist_vel.append(pred_vel_np.copy())
        hist_z.append(z.detach().cpu().numpy().squeeze().copy())
        hist_labels.append(label)

    losses: list[float] = []
    wave_losses: list[float] = []
    reg_losses: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    grad_z_hist: list[float] = []
    grad_wave_hist: list[float] = []
    grad_reg_hist: list[float] = []

    if 0 in snap_times:
        capture("iter 0 (z_init)")

    d = z.numel()
    r_star_sq = float(d - 1)
    reg_min = 0.5 * r_star_sq * (1.0 - math.log(r_star_sq))

    print(f"\n=== Optimizing z  |  opt_steps={opt_steps}  lr={lr}  reg_weight={reg_weight} ===\n", flush=True)

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        pred_n = sample_latent_grad(sampler, z, sigma_max, num_steps, sigma_min, rho, alpha, solver)
        pred_wave = forward_wave(pred_n)
        wave_mse = F.mse_loss(pred_wave, target_wave)
        if reg_weight > 0.0:
            z_norm = torch.norm(z.view(-1), p=2)
            reg = -(d - 1) * torch.log(z_norm + 1e-8) + z_norm ** 2 / 2 - reg_min
            loss = wave_mse + reg_weight * reg
            loss.backward()
            with torch.no_grad():
                r2 = z_norm.item() ** 2 + 1e-16
                grad_reg_vec = reg_weight * z * (1.0 - (d - 1) / r2)
                grad_wave_vec = z.grad - grad_reg_vec
            grad_total = z.grad.norm().item()
            grad_wave_n = grad_wave_vec.norm().item()
            grad_reg_n = grad_reg_vec.norm().item()
            reg_val = (reg_weight * reg).item()
        else:
            loss = wave_mse
            loss.backward()
            grad_total = z.grad.norm().item() if z.grad is not None else 0.0
            grad_wave_n = grad_total
            grad_reg_n = 0.0
            reg_val = 0.0
        opt.step()
        # scheduler.step()

        losses.append(loss.item())
        wave_losses.append(wave_mse.item())
        reg_losses.append(reg_val)
        grad_z_hist.append(grad_total)
        grad_wave_hist.append(grad_wave_n)
        grad_reg_hist.append(grad_reg_n)

        with torch.no_grad():
            pv = v_denormalize_np(pred_n)
            mae_v, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae_v)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            capture(f"iter {it + 1}")

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            if reg_weight > 0.0:
                print(
                    f"iter {it+1}/{opt_steps}  "
                    f"total={loss.item():.6f}  wave={wave_mse.item():.6f}  reg={reg_val:.6f}  "
                    f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  "
                    f"||∇z||={grad_total:.4e}",
                    flush=True,
                )
            else:
                print(
                    f"iter {it+1}/{opt_steps}  wave MSE={loss.item():.6f}  "
                    f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  ||∇z||={grad_total:.4e}",
                    flush=True,
                )

    # --- 图1：优化演化快照（两行：速度场 / 噪声 z）---
    ncols = 2 + len(hist_vel)  # True + smooth + snapshots
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5.5))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    z_hist_max = max((float(np.abs(hz).max()) for hz in hist_z), default=0.0)
    z0lim = max(3.0, float(np.abs(z_init_np).max()) + 0.1, z_hist_max + 0.1)

    # 列 0：真实速度场 + z_init
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("True v", fontsize=8)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(z_init_np, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
    axes[1, 0].set_title("z_init\n(smooth invert)", fontsize=8)
    axes[1, 0].axis("off")

    # 列 1：光滑速度场（可视化参考）
    axes[0, 1].imshow(v_smooth_phys, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 1].set_title(f"v_smooth\n(σ={smooth_sigma})", fontsize=8)
    axes[0, 1].axis("off")
    axes[1, 1].axis("off")

    # 列 2+：优化快照
    for j, (vel, zj, lbl) in enumerate(zip(hist_vel, hist_z, hist_labels)):
        col = j + 2
        axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(lbl, fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("noise z", fontsize=9)
    plt.suptitle(
        f"Wave MSE | EDM PF-ODE steps={num_steps} σ∈[{sigma_min},{sigma_max}] solver={solver} | "
        f"model60[{sample_index}]  z_init: smooth_σ={smooth_sigma} σ_mid={sigma_mid}",
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图2：最终结果（速度 + 波场对比）---
    with torch.no_grad():
        pred_n_final = sample_no_grad(sampler, z.detach(), sigma_max, num_steps, sigma_min, rho, alpha, solver)
        pred_wave_final = forward_wave(pred_n_final)
        pred_vel_np = v_denormalize_np(pred_n_final)

    err_v = pred_vel_np - target_vel_np
    err_lim_v = float(np.max(np.abs(err_v))) + 1e-6
    sh = wave_plot_shot
    tw = target_wave[sh].detach().cpu().numpy()
    pw = pred_wave_final[sh].detach().cpu().numpy()
    ew = pw - tw
    wlim = max(np.abs(tw).max(), np.abs(pw).max()) + 1e-6
    elw = float(np.abs(ew).max()) + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np, "target v (m/s)", "viridis", vel_vmin, vel_vmax),
        (pred_vel_np, "pred v (m/s)", "viridis", vel_vmin, vel_vmax),
        (err_v, "v error (m/s)", "coolwarm", -err_lim_v, err_lim_v),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[1], [
        (tw.T, f"target wave shot {sh}", "seismic", -wlim, wlim),
        (pw.T, f"pred wave shot {sh}", "seismic", -wlim, wlim),
        (ew.T, "wave residual", "seismic", -elw, elw),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    reg_str = f"  reg={reg_losses[-1]:.6f}" if reg_weight > 0.0 else ""
    plt.suptitle(
        f"Final | wave MSE={wave_losses[-1]:.6f}{reg_str}  total={losses[-1]:.6f}  "
        f"vel MAE={mae_hist[-1]:.1f} m/s  SSIM={ssim_hist[-1]:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图3：指标曲线 ---
    iters = list(range(1, len(losses) + 1))
    use_reg = reg_weight > 0.0
    if use_reg:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        axes[0, 0].plot(iters, wave_losses, color="C0")
        axes[0, 0].set_ylabel("wave MSE")
        axes[0, 0].set_title("Data loss (wave MSE)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(iters, reg_losses, color="C5")
        axes[0, 1].set_ylabel(f"reg_weight × reg  (λ={reg_weight})")
        axes[0, 1].set_title("Regularization loss (χ-NLL scaled)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(iters, losses, color="C0", label="total")
        axes[1, 0].plot(iters, wave_losses, color="C0", linestyle="--", alpha=0.5, label="wave MSE")
        axes[1, 0].plot(iters, reg_losses, color="C5", linestyle="--", alpha=0.5, label="reg")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Total loss")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(iters, mae_hist, color="C1")
        axes[1, 1].set_ylabel("MAE (m/s)")
        axes[1, 1].set_title("Velocity MAE")
        axes[1, 1].grid(True, alpha=0.3)
        axes[2, 0].semilogy(iters, ssim_hist, color="C2")
        axes[2, 0].set_ylabel("SSIM")
        axes[2, 0].set_xlabel("Iteration")
        axes[2, 0].set_title("Velocity SSIM")
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 1].semilogy(iters, grad_z_hist, color="C3", label="||∇z|| total")
        axes[2, 1].semilogy(iters, grad_wave_hist, color="C0", linestyle="--", alpha=0.8, label="||∇z||_wave")
        axes[2, 1].semilogy(iters, grad_reg_hist, color="C5", linestyle="--", alpha=0.8, label="||∇z||_reg")
        axes[2, 1].set_ylabel("||∇z|| (log)")
        axes[2, 1].set_xlabel("Iteration")
        axes[2, 1].set_title("Gradient norm decomposition")
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
        axes[0, 0].plot(iters, losses, color="C0")
        axes[0, 0].set_ylabel("wave MSE")
        axes[0, 0].set_title("Wave-space loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(iters, mae_hist, color="C1")
        axes[0, 1].set_ylabel("MAE (m/s)")
        axes[0, 1].set_title("Velocity MAE")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(iters, ssim_hist, color="C2")
        axes[1, 0].set_ylabel("SSIM")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_title("Velocity SSIM")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].semilogy(iters, grad_z_hist, color="C3")
        axes[1, 1].set_ylabel("||∇z|| (log scale)")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_title("Gradient norm of z")
        axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle("Optimization metrics", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "final_total_loss": float(losses[-1]),
        "final_wave_mse": float(wave_losses[-1]),
        "final_reg_loss": float(reg_losses[-1]),
        "final_vel_mae_m_s": float(mae_hist[-1]),
        "final_vel_ssim": float(ssim_hist[-1]),
        "reg_weight": reg_weight,
        "z_init": {
            "smooth_sigma": smooth_sigma,
            "sigma_mid": sigma_mid,
            "mae_smooth_m_s": mae_smooth,
            "ssim_smooth": ssim_smooth,
            "mae_ode_zinit_m_s": mae_init,
            "ssim_ode_zinit": ssim_init,
        },
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
