#!/usr/bin/env python3
"""
预训练 EDM（``EDMPrecond``）+ ``edm_sampler_ode`` / ``edm_sampler_ode_latentgrad``：
优化标准高斯噪声 z，使 PF-ODE 得到的速度经 ``seismic_master_forward_modeling``
与目标波场 MSE 最小（需 CUDA + CuPy）。

初值约定：``latents = z * sigma_max``。

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/edm_opt_wave/demo_edm_opt_wave.py \\
    --config exp/edm_opt_wave/config_edm_opt_wave.yaml
"""

from __future__ import annotations

import argparse
import json
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]  # exp/edm_opt_wave -> exp -> Manifold_constrained_FWI

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
from src.core.generate import edm_sampler_ode, edm_sampler_ode_latentgrad
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
    """无梯度快照采样（用于可视化）。"""
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
    """带梯度优化采样。"""
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
    """EDM 输出 [-1,1]（70×70）→ 物理速度 → 正演波场 (5, 1000, 70)。"""
    v_phys = v_denormalize_tensor(pred_norm.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


def load_target_velocity(npy_path: Path, sample_index: int) -> np.ndarray:
    data = np.load(npy_path)  # (N, 1, 70, 70)
    return data[sample_index, 0].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="EDM 初始噪声优化（波场 MSE）")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_edm_opt_wave.yaml"),
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

    cv = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    data60_path = model60_path.parent / model60_path.name.replace("model", "data")
    sample_index = int(cv.get("sample_index", 0))

    opt_cfg = cfg.get("optimization") or {}
    seed_z = int(opt_cfg.get("seed_z", 42))
    opt_steps = int(opt_cfg.get("opt_steps", 100))
    lr = float(opt_cfg.get("lr", 0.05))
    snapshots = int(opt_cfg.get("snapshots", 10))

    viz = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"edm_opt_wave_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
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
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    grad_z_hist: list[float] = []

    if 0 in snap_times:
        capture("iter 0 (init)")

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        pred_n = sample_latent_grad(sampler, z, sigma_max, num_steps, sigma_min, rho, alpha, solver)
        pred_wave = forward_wave(pred_n)
        loss = F.mse_loss(pred_wave, target_wave)
        loss.backward()
        grad_z_hist.append(z.grad.norm().item() if z.grad is not None else 0.0)
        opt.step()
        scheduler.step()

        losses.append(loss.item())
        with torch.no_grad():
            pv = v_denormalize_np(pred_n)
            mae_v, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae_v)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            capture(f"iter {it + 1}")

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            print(
                f"iter {it + 1}/{opt_steps}  wave MSE={loss.item():.6f}  "
                f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  ||∇z||={grad_z_hist[-1]:.4e}"
            )

    # --- 图1：优化演化快照 ---
    ncols = 1 + len(hist_vel)
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5.5))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    z_hist_max = max((float(np.abs(hz).max()) for hz in hist_z), default=0.0)
    z0lim = max(3.0, float(np.abs(z_init_np).max()) + 0.1, z_hist_max + 0.1)
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("target v\n(model60)", fontsize=8)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(z_init_np, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
    axes[1, 0].set_title("init z", fontsize=8)
    axes[1, 0].axis("off")
    for j, (vel, zj, lbl) in enumerate(zip(hist_vel, hist_z, hist_labels)):
        axes[0, j + 1].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, j + 1].set_title(lbl, fontsize=8)
        axes[0, j + 1].axis("off")
        axes[1, j + 1].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
        axes[1, j + 1].axis("off")
    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("noise z", fontsize=9)
    plt.suptitle(
        f"Wave MSE | EDM PF-ODE steps={num_steps} σ∈[{sigma_min},{sigma_max}] solver={solver} | "
        f"model60[{sample_index}] seed_z={seed_z}",
        fontsize=9,
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
    plt.suptitle(
        f"Final | wave MSE={losses[-1]:.6f}  vel MAE={mae_hist[-1]:.1f} m/s  SSIM={ssim_hist[-1]:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图3：指标曲线（wave Loss / MAE / SSIM / ||∇z||）---
    iters = list(range(1, len(losses) + 1))
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
        "final_wave_mse": float(losses[-1]),
        "final_vel_mae_m_s": float(mae_hist[-1]),
        "final_vel_ssim": float(ssim_hist[-1]),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
