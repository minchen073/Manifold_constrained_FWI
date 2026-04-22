#!/usr/bin/env python3
"""
预训练 DDIM（``OpenFWIUNetWrapper`` + ``DDIMScheduler``）：
优化标准高斯噪声 z，使 DDIM 得到的速度经 ``seismic_master_forward_modeling``
与目标波场 MSE 最小（需 CUDA + CuPy）。

初值约定：z ~ N(0, I)，shape (1,1,70,70)，直接作为 DDIM 的 x_T。
（与 EDM 的 ``latents = z * sigma_max`` 不同，DDIM 无需缩放。）

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/DDIM/edm_opt_wave/demo_ddim_opt_wave.py \\
    --config exp/DDIM/edm_opt_wave/config_ddim_opt_wave.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]  # exp/DDIM/edm_opt_wave -> DDIM -> exp -> Manifold_constrained_FWI
_TRAINING_DIR = _MANIFOLD_ROOT / "training"
_CURVE_VEL_B = _MANIFOLD_ROOT.parent / "CurveVelB"

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
    """[-1, 1] → 物理速度 (m/s)，保持计算图。"""
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


def load_ddim_pipeline(checkpoint: Path, training_yaml: Path | None):
    """加载 OpenFWIUNetWrapper + DDIMScheduler（clip_sample=False 以保证梯度流通）。

    返回 (wrapper, ddim_sched)。
    """
    checkpoint = checkpoint.resolve()
    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}; expected training checkpoint (torch.save of OpenFWIUNetWrapper).")

    torch_dtype = torch.float32
    if training_yaml is not None and training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s = str((cfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        torch_dtype = torch.float16 if s in ("float16", "fp16") else torch.float32

    wrapper = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)

    ddpm_sched = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    # clip_sample=False：避免梯度在 clamp 边界被截断，保证优化梯度信号畅通
    # from_config 会自动过滤 DDPMScheduler 独有的字段（如 variance_type）
    ddim_sched = DDIMScheduler.from_config(ddpm_sched.config, clip_sample=False)

    return wrapper, ddim_sched


def sample_with_grad(
    wrapper, ddim_sched, z: torch.Tensor, num_steps: int, eta: float = 0.0
) -> torch.Tensor:
    """带梯度 DDIM 采样。z: (1,1,70,70)，返回 (70,70) in [-1,1]。

    不加 @torch.no_grad()，计算图完整保留，梯度可回传到 z。
    """
    ddim_sched.set_timesteps(num_steps)
    image = z  # (1, 1, 70, 70)
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


def forward_wave(pred_norm: torch.Tensor) -> torch.Tensor:
    """DDIM 输出 [-1,1]（70×70）→ 物理速度 → 正演波场 (5, 1000, 70)。"""
    v_phys = v_denormalize_tensor(pred_norm.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


def load_target_velocity(npy_path: Path, sample_index: int) -> np.ndarray:
    data = np.load(npy_path)  # (N, 1, 70, 70)
    return data[sample_index, 0].astype(np.float32)


def plot_perturbation_grid(
    z_np: np.ndarray,
    wrapper,
    ddim_sched,
    num_steps: int,
    eta: float,
    device: torch.device,
    vel_vmin: float,
    vel_vmax: float,
    out_path: Path,
    label: str = "",
    perturb_std: float = 0.01,
    seed: int = 0,
) -> None:
    """对输入噪声 z 施加 9 个独立的高斯微扰，将 DDIM 解码的速度场绘成 3×3 九宫格。

    Args:
        z_np:        基准噪声，shape (70,70) 或 (1,1,70,70)，numpy float32。
        perturb_std: 微扰幅度（默认 0.01，量级远小于 z ~ N(0,1)）。
        label:       图标题中显示的标签，如 "init z" / "final z"。
        seed:        生成 9 组微扰的随机种子（保证可复现）。
    """
    z_base = torch.from_numpy(z_np.astype(np.float32)).reshape(1, 1, 70, 70).to(device)

    rng = torch.Generator(device=device).manual_seed(seed)
    vels: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(9):
            eps = torch.randn(z_base.shape, device=device, generator=rng) * perturb_std
            pred_n = sample_no_grad(wrapper, ddim_sched, z_base + eps, num_steps, eta)
            vels.append(v_denormalize_np(pred_n))

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
    for idx, (ax, vel) in enumerate(zip(axes.ravel(), vels)):
        im = ax.imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        ax.set_title(f"perturb #{idx + 1}", fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="v (m/s)")
    plt.suptitle(
        f"Perturbation neighborhood  |  {label}\n"
        f"z + N(0, {perturb_std}²·I),  DDIM steps={num_steps}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM 初始噪声优化（波场 MSE）")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_opt_wave.yaml"),
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

    # --- 加载 DDIM pipeline ---
    dcfg = cfg.get("ddim") or {}
    ckpt = _resolve_path(dcfg["checkpoint"])
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    training_yaml_arg = dcfg.get("training_yaml")
    if training_yaml_arg:
        training_yaml = _resolve_path(training_yaml_arg)
    else:
        candidate = (ckpt.parent / "config_used.yaml").resolve()
        training_yaml = candidate if candidate.is_file() else (
            _TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml"
        ).resolve()

    wrapper, ddim_sched = load_ddim_pipeline(ckpt, training_yaml)
    wrapper = wrapper.to(device)
    wrapper.eval()

    num_steps = int(dcfg.get("num_steps", 20))
    eta = float(dcfg.get("eta", 0.0))

    cv = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    sample_index = int(cv.get("sample_index", 0))

    opt_cfg = cfg.get("optimization") or {}
    seed_z = int(opt_cfg.get("seed_z", 42))
    opt_steps = int(opt_cfg.get("opt_steps", 100))
    lr = float(opt_cfg.get("lr", 0.05))
    snapshots = int(opt_cfg.get("snapshots", 10))
    reg_weight = float(opt_cfg.get("reg_weight", 0.0))

    viz = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_opt_wave_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "checkpoint": str(ckpt),
        "training_yaml": str(training_yaml),
        "model60_path": str(model60_path),
        "output_dir": str(out_dir),
        "device_used": str(device),
        "config_path": str(cfg_path.resolve()),
        "manifold_root": str(_MANIFOLD_ROOT),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    target_vel_np = load_target_velocity(model60_path, sample_index)

    # 目标波场：直接由真值速度正演得到
    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

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
        pred_vel_np = v_denormalize_np(pred_n)
        hist_vel.append(pred_vel_np.copy())
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
    wave_losses: list[float] = []
    reg_losses: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    grad_z_hist: list[float] = []
    grad_wave_hist: list[float] = []
    grad_reg_hist: list[float] = []

    if 0 in snap_times:
        capture("iter 0 (init)")

    d = z.numel()  # = 1×1×70×70 = 4900
    # χ(d) NLL 在 r* = √(d-1) 处的最小值（用于归一化使 reg ≥ 0）
    r_star_sq = float(d - 1)
    reg_min = 0.5 * r_star_sq * (1.0 - math.log(r_star_sq))

    print(
        f"[DDIM opt_wave] checkpoint={ckpt.name}  steps={num_steps}  eta={eta}  "
        f"opt_steps={opt_steps}  lr={lr}  reg_weight={reg_weight}  "
        f"sample_index={sample_index}  seed_z={seed_z}  device={device}"
    )

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        pred_n = sample_with_grad(wrapper, ddim_sched, z, num_steps, eta)
        pred_n.retain_grad()
        pred_wave = forward_wave(pred_n)
        wave_mse = F.mse_loss(pred_wave, target_wave)
        if reg_weight > 0.0:
            z_norm = torch.norm(z.view(-1), p=2)
            # χ(d) NLL（已减去最小值，使 reg ≥ 0，最小值在 ||z|| = √(d-1) ≈ 70 处取 0）
            reg = -(d - 1) * torch.log(z_norm + 1e-8) + z_norm ** 2 / 2 - reg_min
            loss = wave_mse + reg_weight * reg
            loss.backward()
            # 解析计算 reg 对 z 的梯度：∂(reg_weight·reg)/∂z = reg_weight·z·(1 - (d-1)/r²)
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
        # lr_sched.step()

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
            gv = pred_n.grad.detach().cpu().numpy().squeeze().copy() if pred_n.grad is not None else None
            capture(f"iter {it + 1}", pred_n_grad_np=gv)

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            if reg_weight > 0.0:
                print(
                    f"iter {it+1}/{opt_steps}  "
                    f"total={loss.item():.6f}  wave={wave_mse.item():.6f}  reg={reg_val:.6f}  "
                    f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  "
                    f"||∇z||={grad_total:.4e}  ||∇z||_wave={grad_wave_n:.4e}  ||∇z||_reg={grad_reg_n:.4e}"
                )
            else:
                print(
                    f"iter {it + 1}/{opt_steps}  wave MSE={loss.item():.6f}  "
                    f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  ||∇z||={grad_z_hist[-1]:.4e}"
                )

    # --- 图1：优化演化快照（速度场 + z + ∇z + ∇pred_n）---
    ncols = 1 + len(hist_vel)
    fig, axes = plt.subplots(4, ncols, figsize=(2.6 * ncols, 10.0))
    axes = np.atleast_2d(axes).reshape(4, ncols)
    z_hist_max = max((float(np.abs(hz).max()) for hz in hist_z), default=0.0)
    z0lim = max(3.0, float(np.abs(z_init_np).max()) + 0.1, z_hist_max + 0.1)
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("target v\n(model60)", fontsize=8)
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

    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("noise z", fontsize=9)
    axes[2, 0].set_ylabel("∇z (grad)", fontsize=9)
    axes[3, 0].set_ylabel("∇x̂₀ (pred grad)", fontsize=9)
    plt.suptitle(
        f"Wave MSE | DDIM steps={num_steps} eta={eta} | "
        f"model60[{sample_index}] seed_z={seed_z}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图2：最终结果（速度 + 波场对比）---
    with torch.no_grad():
        pred_n_final = sample_no_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)
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
        (pred_vel_np, "pred v (m/s) [DDIM]", "viridis", vel_vmin, vel_vmax),
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

    # --- 图3：指标曲线（wave Loss / MAE / SSIM / ||∇z||）---
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
        axes[1, 0].set_title("Total loss (wave MSE + reg)")
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

    # --- 图4 & 5：微扰邻域九宫格（初始噪声 vs 最终噪声）---
    perturb_std = float(viz.get("perturb_std", 0.01))
    plot_perturbation_grid(
        z_init_np, wrapper, ddim_sched, num_steps, eta, device,
        vel_vmin, vel_vmax,
        out_path=out_dir / "perturb_grid_init.png",
        label=f"init z  (seed_z={seed_z})",
        perturb_std=perturb_std,
        seed=0,
    )
    plot_perturbation_grid(
        z.detach().cpu().numpy().squeeze(), wrapper, ddim_sched, num_steps, eta, device,
        vel_vmin, vel_vmax,
        out_path=out_dir / "perturb_grid_final.png",
        label=f"final z  (after {opt_steps} iters)",
        perturb_std=perturb_std,
        seed=0,
    )

    summary = {
        "final_total_loss": float(losses[-1]),
        "final_wave_mse": float(wave_losses[-1]),
        "final_reg_loss": float(reg_losses[-1]),
        "final_vel_mae_m_s": float(mae_hist[-1]),
        "final_vel_ssim": float(ssim_hist[-1]),
        "reg_weight": reg_weight,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
