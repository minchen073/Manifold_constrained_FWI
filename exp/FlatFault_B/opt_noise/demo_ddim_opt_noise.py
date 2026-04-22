#!/usr/bin/env python3
"""
预训练 DDIM（``OpenFWIUNetWrapper`` + ``DDIMScheduler``）—— FlatFault-B 版本

优化标准高斯噪声 z，使 DDIM 采样速度与目标在归一化 [-1,1] 空间 MSE 最小。

支持两种模型加载方式：
  1. checkpoint 目录（含 model.pt + scheduler/ 子目录）—— 兼容训练 run
  2. 独立 model.pt 文件（pretrained_model/FlatFault_DDIM/model.pt）—— scheduler
     参数由 config 中的 scheduler_config 字段指定

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/FlatFault_B/edm_opt_noise/demo_ddim_opt_noise.py \\
    --config exp/FlatFault_B/edm_opt_noise/config_ddim_opt_noise.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
# exp/FlatFault_B/edm_opt_noise → FlatFault_B → exp → Manifold_constrained_FWI
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR = _MANIFOLD_ROOT / "training"
_CURVE_VEL_B = _MANIFOLD_ROOT.parent / "CurveVelB"  # diffusers_torch_compat 所在目录

for _p in [str(_CURVE_VEL_B), str(_TRAINING_DIR), str(_MANIFOLD_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Suppress C-level stderr during diffusers import (xformers / flash-attn custom ops)
_null_fd = os.open(os.devnull, os.O_WRONLY)
_saved_fd2 = os.dup(2)
_saved_py_stderr = sys.stderr
os.dup2(_null_fd, 2)
os.close(_null_fd)
sys.stderr = open(os.devnull, "w")
try:
    from diffusers_torch_compat import ensure_diffusers_custom_ops_safe
    ensure_diffusers_custom_ops_safe()
    from diffusers import DDIMScheduler, DDPMScheduler
    from openfwi_unet_wrapper import load_openfwi_checkpoint
finally:
    sys.stderr.close()
    sys.stderr = _saved_py_stderr
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd2)

from src.core import pytorch_ssim


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def v_normalize(v: torch.Tensor | np.ndarray) -> torch.Tensor:
    """物理速度 (m/s) → 归一化 [-1, 1]：x = (v − 3000) / 1500。"""
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v.astype(np.float32))
    return (v - 3000.0) / 1500.0


def v_denormalize_np(v_norm: torch.Tensor | np.ndarray) -> np.ndarray:
    """[-1, 1] → 物理速度 (m/s)，返回 numpy。"""
    if isinstance(v_norm, torch.Tensor):
        x = v_norm.detach().float().cpu().numpy()
    else:
        x = v_norm.astype(np.float32)
    return x * 1500.0 + 3000.0


def vel_to_ssim_tensor(vel_np: np.ndarray, vmin: float, vmax: float) -> torch.Tensor:
    t = torch.from_numpy(vel_np.astype(np.float32)).view(1, 1, vel_np.shape[0], vel_np.shape[1])
    return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0)


def load_target_velocity(npy_path: Path, sample_index: int, device: torch.device):
    """返回 (target_vel_np HxW m/s, target_norm (1,1,70,70) 归一化 on device)。"""
    data = np.load(npy_path)  # (N, 1, 70, 70)
    v_np = data[sample_index, 0].astype(np.float32)
    target_norm = v_normalize(v_np).to(device).view(1, 1, 70, 70)
    return v_np, target_norm


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


def load_ddim_pipeline(model_path: Path, sched_cfg: dict | None = None):
    """加载 OpenFWIUNetWrapper + DDIMScheduler（clip_sample=False）。

    model_path 可以是：
      - 包含 model.pt + scheduler/ 的 checkpoint 目录
      - 直接指向独立 model.pt 文件（此时 scheduler 由 sched_cfg 构建）

    返回 (wrapper, ddim_sched)。
    """
    model_path = model_path.resolve()

    # 判断是目录还是直接 .pt 文件
    if model_path.is_dir():
        model_pt = model_path / "model.pt"
        if not model_pt.is_file():
            raise SystemExit(f"Missing {model_pt}")
        ddpm_sched = DDPMScheduler.from_pretrained(str(model_path), subfolder="scheduler")
        ddim_sched = DDIMScheduler.from_config(ddpm_sched.config, clip_sample=False)
        wrapper = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch.float32)
    elif model_path.is_file() and model_path.suffix == ".pt":
        if sched_cfg is None:
            sched_cfg = {}
        ddpm_sched = DDPMScheduler(
            num_train_timesteps=int(sched_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(sched_cfg.get("beta_start", 0.0001)),
            beta_end=float(sched_cfg.get("beta_end", 0.02)),
            beta_schedule=sched_cfg.get("beta_schedule", "linear"),
            prediction_type=sched_cfg.get("prediction_type", "epsilon"),
            timestep_spacing=sched_cfg.get("timestep_spacing", "leading"),
            clip_sample=False,
        )
        ddim_sched = DDIMScheduler.from_config(ddpm_sched.config, clip_sample=False)
        wrapper = load_openfwi_checkpoint(model_path, map_location="cpu", torch_dtype=torch.float32)
    else:
        raise SystemExit(f"model_path 不存在或格式不支持: {model_path}")

    return wrapper, ddim_sched


def sample_with_grad(
    wrapper, ddim_sched, z: torch.Tensor, num_steps: int, eta: float = 0.0
) -> torch.Tensor:
    """带梯度 DDIM 采样。z: (1,1,70,70)，返回 (70,70) in [-1,1]。"""
    ddim_sched.set_timesteps(num_steps)
    image = z
    for t in ddim_sched.timesteps:
        model_output = wrapper(image, t).sample
        image = ddim_sched.step(
            model_output, t, image, eta=eta, use_clipped_model_output=False
        ).prev_sample
    return image.squeeze()  # (70, 70)


def sample_no_grad(
    wrapper, ddim_sched, z: torch.Tensor, num_steps: int, eta: float = 0.0
) -> torch.Tensor:
    """无梯度快照采样（用于可视化）。"""
    with torch.no_grad():
        return sample_with_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM 初始噪声优化（FlatFault-B）")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_opt_noise.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path.resolve())

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 加载 DDIM pipeline ---
    dcfg = cfg.get("ddim") or {}
    model_path = _resolve_path(dcfg["model_path"])
    sched_cfg = dcfg.get("scheduler_config") or {}

    num_steps = int(dcfg.get("num_steps", 20))
    eta = float(dcfg.get("eta", 0.0))

    wrapper, ddim_sched = load_ddim_pipeline(model_path, sched_cfg)
    wrapper = wrapper.to(device)
    wrapper.eval()

    # --- FlatFault-B 数据 ---
    ff_cfg = cfg.get("flatfault") or {}
    data_path = _resolve_path(ff_cfg["data_path"])
    sample_index = int(ff_cfg.get("sample_index", 0))

    opt_cfg = cfg.get("optimization") or {}
    seed_z = int(opt_cfg.get("seed_z", 42))
    opt_steps = int(opt_cfg.get("opt_steps", 300))
    lr = float(opt_cfg.get("lr", 0.02))
    snapshots = int(opt_cfg.get("snapshots", 10))
    reg_weight = float(opt_cfg.get("reg_weight", 0.0))

    viz = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))

    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_opt_noise_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "output_dir": str(out_dir),
        "device_used": str(device),
        "config_path": str(cfg_path.resolve()),
        "manifold_root": str(_MANIFOLD_ROOT),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    if not data_path.is_file():
        raise SystemExit(f"Data not found: {data_path}")
    target_vel_np, target_norm = load_target_velocity(data_path, sample_index, device)

    g_z = torch.Generator(device=device).manual_seed(seed_z)
    z = torch.randn(1, 1, 70, 70, device=device, dtype=torch.float32, generator=g_z, requires_grad=True)
    z_init_np = z.detach().cpu().numpy().squeeze().copy()

    opt = torch.optim.Adam([z], lr=lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=0.0)
    snap_times = np.unique(np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int))
    snap_times = np.clip(snap_times, 0, opt_steps)

    hist_vel:       list[np.ndarray] = []
    hist_z:         list[np.ndarray] = []
    hist_grad:      list[np.ndarray] = []
    hist_grad_vel:  list[np.ndarray] = []
    hist_labels:    list[str] = []

    def capture(label: str, pred_n_grad_np: np.ndarray | None = None) -> None:
        pred_n = sample_no_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)
        pred_vel = v_denormalize_np(pred_n)
        hist_vel.append(pred_vel)
        hist_z.append(z.detach().cpu().numpy().squeeze().copy())
        hist_labels.append(label)
        grad_np = (
            z.grad.detach().cpu().numpy().squeeze().copy()
            if z.grad is not None
            else np.zeros(z.shape[-2:], dtype=np.float32)
        )
        hist_grad.append(grad_np)
        hist_grad_vel.append(
            pred_n_grad_np if pred_n_grad_np is not None
            else np.zeros(z.shape[-2:], dtype=np.float32)
        )

    losses: list[float] = []
    mse_losses: list[float] = []
    reg_losses: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    grad_total_hist: list[float] = []
    grad_mse_hist: list[float] = []
    grad_reg_hist: list[float] = []

    if 0 in snap_times:
        capture("iter 0 (init)")

    d = z.numel()  # = 1×1×70×70 = 4900

    data_file = data_path.name
    print(
        f"[DDIM opt_noise | FlatFault-B] model={model_path.name}  "
        f"data={data_file}[{sample_index}]  steps={num_steps}  eta={eta}  "
        f"opt_steps={opt_steps}  lr={lr}  reg_weight={reg_weight}  "
        f"seed_z={seed_z}  device={device}"
    )

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        pred_n = sample_with_grad(wrapper, ddim_sched, z, num_steps, eta)
        pred_n.retain_grad()
        pred_b = pred_n.view(1, 1, 70, 70)
        mse = F.mse_loss(pred_b, target_norm)
        if reg_weight > 0.0:
            z_norm = torch.norm(z.view(-1), p=2)
            # χ(d) NLL: -(d-1)·log(r) + r²/2  (minimum at r* = √(d-1) ≈ 70)
            reg = -(d - 1) * torch.log(z_norm + 1e-8) + z_norm ** 2 / 2
            loss = mse + reg_weight * reg
            loss.backward()
            with torch.no_grad():
                r2 = z_norm.item() ** 2 + 1e-16
                grad_reg_vec = reg_weight * z * (1.0 - (d - 1) / r2)
                grad_mse_vec = z.grad - grad_reg_vec
            grad_total = z.grad.norm().item()
            grad_mse_n = grad_mse_vec.norm().item()
            grad_reg_n = grad_reg_vec.norm().item()
            reg_val = (reg_weight * reg).item()
        else:
            loss = mse
            loss.backward()
            grad_total = z.grad.norm().item() if z.grad is not None else 0.0
            grad_mse_n = grad_total
            grad_reg_n = 0.0
            reg_val = 0.0
        opt.step()
        lr_sched.step()

        losses.append(loss.item())
        mse_losses.append(mse.item())
        reg_losses.append(reg_val)
        grad_total_hist.append(grad_total)
        grad_mse_hist.append(grad_mse_n)
        grad_reg_hist.append(grad_reg_n)

        with torch.no_grad():
            pv = v_denormalize_np(pred_n)
            mae, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            gv = pred_n.grad.detach().cpu().numpy().squeeze().copy() if pred_n.grad is not None else None
            capture(f"iter {it + 1}", pred_n_grad_np=gv)

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            if reg_weight > 0.0:
                print(
                    f"iter {it+1}/{opt_steps}  "
                    f"total={loss.item():.6f}  MSE={mse.item():.6f}  reg={reg_val:.6f}  "
                    f"MAE={mae:.1f} m/s  SSIM={ssim_v:.4f}  "
                    f"||∇z||={grad_total:.4e}  ||∇z||_mse={grad_mse_n:.4e}  ||∇z||_reg={grad_reg_n:.4e}"
                )
            else:
                print(
                    f"iter {it+1}/{opt_steps}  MSE={mse.item():.6f}  "
                    f"MAE={mae:.1f} m/s  SSIM={ssim_v:.4f}  ||∇z||={grad_total:.4e}"
                )

    # --- 图1：优化演化快照 ---
    ncols = 1 + len(hist_vel)
    fig, axes = plt.subplots(4, ncols, figsize=(2.6 * ncols, 10.0))
    axes = np.atleast_2d(axes).reshape(4, ncols)
    z_hist_max = max((float(np.abs(hz).max()) for hz in hist_z), default=0.0)
    z0lim = max(3.0, float(np.abs(z_init_np).max()) + 0.1, z_hist_max + 0.1)
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title(f"target v\n{data_file}[{sample_index}]", fontsize=8)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(z_init_np, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
    axes[1, 0].set_title("init z", fontsize=8)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[3, 0].axis("off")

    for j, (vel, zj, gj, gvj, lbl) in enumerate(zip(hist_vel, hist_z, hist_grad, hist_grad_vel, hist_labels)):
        col = j + 1
        axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(lbl, fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
        axes[1, col].axis("off")
        glim = max(float(np.percentile(np.abs(gj), 99)), 1e-8)
        axes[2, col].imshow(gj, cmap="coolwarm", aspect="auto", vmin=-glim, vmax=glim)
        axes[2, col].axis("off")
        gvlim = max(float(np.percentile(np.abs(gvj), 99)), 1e-8)
        axes[3, col].imshow(gvj, cmap="coolwarm", aspect="auto", vmin=-gvlim, vmax=gvlim)
        axes[3, col].axis("off")

    axes[0, 0].set_ylabel("velocity (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("noise z", fontsize=9)
    axes[2, 0].set_ylabel("∇z (grad)", fontsize=9)
    axes[3, 0].set_ylabel("∇x̂₀ (pred grad)", fontsize=9)
    plt.suptitle(
        f"FlatFault-B DDIM | steps={num_steps} eta={eta} | {data_file}[{sample_index}] seed_z={seed_z}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图2：最终结果 ---
    pred_n_final = sample_no_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)
    pred_vel_np = v_denormalize_np(pred_n_final)
    err = pred_vel_np - target_vel_np
    err_lim = float(np.max(np.abs(err))) + 1e-6
    z_opt_np = z.detach().cpu().numpy().squeeze()
    z_noise_lim = max(float(np.abs(z_opt_np).max()), float(np.abs(z_init_np).max())) + 0.1
    dz = z_opt_np - z_init_np
    dz_lim = float(np.abs(dz).max()) + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np, f"target v  {data_file}[{sample_index}]", "viridis", vel_vmin, vel_vmax),
        (pred_vel_np,   "optimized v (DDIM)", "viridis", vel_vmin, vel_vmax),
        (err,           "error (m/s)", "coolwarm", -err_lim, err_lim),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[1], [
        (z_init_np, "z_init",        "coolwarm", -z_noise_lim, z_noise_lim),
        (z_opt_np,  "z after optim", "coolwarm", -z_noise_lim, z_noise_lim),
        (dz,        "z_opt − z_init","coolwarm", -dz_lim,      dz_lim),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    reg_str = f"  reg={reg_losses[-1]:.6f}" if reg_weight > 0.0 else ""
    plt.suptitle(
        f"Final | MAE={mae_hist[-1]:.1f} m/s  SSIM={ssim_hist[-1]:.4f}  "
        f"MSE={mse_losses[-1]:.6f}{reg_str}  total={losses[-1]:.6f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图3：指标曲线 ---
    iters = list(range(1, len(losses) + 1))
    use_reg = reg_weight > 0.0
    if use_reg:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        axes[0, 0].plot(iters, mse_losses, color="C0")
        axes[0, 0].set_ylabel("MSE [-1,1]")
        axes[0, 0].set_title("Data loss (MSE)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(iters, reg_losses, color="C5")
        axes[0, 1].set_ylabel(f"reg_weight × reg  (λ={reg_weight})")
        axes[0, 1].set_title("Regularization loss (χ-NLL scaled)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(iters, losses, color="C0", label="total")
        axes[1, 0].plot(iters, mse_losses, color="C0", linestyle="--", alpha=0.5, label="MSE")
        axes[1, 0].plot(iters, reg_losses, color="C5", linestyle="--", alpha=0.5, label="reg")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Total loss (MSE + reg)")
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
        axes[2, 1].semilogy(iters, grad_total_hist, color="C3", label="||∇z|| total")
        axes[2, 1].semilogy(iters, grad_mse_hist, color="C0", linestyle="--", alpha=0.8, label="||∇z||_mse")
        axes[2, 1].semilogy(iters, grad_reg_hist, color="C5", linestyle="--", alpha=0.8, label="||∇z||_reg")
        axes[2, 1].set_ylabel("||∇z|| (log)")
        axes[2, 1].set_xlabel("Iteration")
        axes[2, 1].set_title("Gradient norm decomposition")
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
        axes[0, 0].plot(iters, mse_losses, color="C0")
        axes[0, 0].set_ylabel("MSE [-1,1]")
        axes[0, 0].set_title("Image-space loss")
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
        axes[1, 1].semilogy(iters, grad_total_hist, color="C3")
        axes[1, 1].set_ylabel("||∇z|| (log scale)")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_title("Gradient norm of z")
        axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle("Optimization metrics  (FlatFault-B)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "final_total_loss":  float(losses[-1]),
        "final_mse_norm":    float(mse_losses[-1]),
        "final_reg_loss":    float(reg_losses[-1]),
        "final_mae_m_s":     float(mae_hist[-1]),
        "final_ssim":        float(ssim_hist[-1]),
        "reg_weight":        reg_weight,
        "data_file":         str(data_path),
        "sample_index":      sample_index,
        "out_dir":           str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
