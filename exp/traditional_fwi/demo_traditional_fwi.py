#!/usr/bin/env python3
"""
传统全波形反演（FWI）：直接在归一化速度空间 v ∈ [-1,1] 优化，
使 seismic_master_forward_modeling 正演波场与目标波场 MSE 最小（需 CUDA + CuPy）。

初值约定：v 归一化为 [-1,1]（μ = (v_phys − 3000) / 1500）。

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/traditional_fwi/demo_traditional_fwi.py \
    --config exp/traditional_fwi/config_traditional_fwi.yaml
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
from scipy.ndimage import gaussian_filter

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]  # exp/traditional_fwi -> exp -> Manifold_constrained_FWI

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

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


def tv_loss(v: torch.Tensor) -> torch.Tensor:
    """各向同性 TV：mean(|∂v/∂x| + |∂v/∂y|)。"""
    dh = (v[1:, :] - v[:-1, :]).abs()
    dw = (v[:, 1:] - v[:, :-1]).abs()
    return dh.mean() + dw.mean()


def tikhonov_loss(v: torch.Tensor) -> torch.Tensor:
    """Tikhonov 平滑正则：mean((∂v/∂x)² + (∂v/∂y)²)。"""
    dh = v[1:, :] - v[:-1, :]
    dw = v[:, 1:] - v[:, :-1]
    return (dh ** 2).mean() + (dw ** 2).mean()


def build_init_velocity(target_vel_np: np.ndarray, method: str, device: torch.device,
                        smooth_sigma: float = 10.0) -> torch.Tensor:
    """构造初始归一化速度场（shape 70×70）。"""
    if method == "smooth":
        v_smooth = gaussian_filter(target_vel_np.astype(np.float32), sigma=smooth_sigma)
        v_norm = (v_smooth - 3000.0) / 1500.0
        return torch.from_numpy(v_norm).to(device)
    elif method == "constant":
        mean_m_s = float(target_vel_np.mean())
        v_norm = (mean_m_s - 3000.0) / 1500.0
        return torch.full((70, 70), v_norm, dtype=torch.float32, device=device)
    elif method == "gaussian":
        return torch.randn(70, 70, dtype=torch.float32, device=device) * 0.1
    else:
        raise ValueError(f"Unknown init method: {method!r}. Choose smooth | constant | gaussian.")


def main() -> None:
    parser = argparse.ArgumentParser(description="传统 FWI 速度优化（波场 MSE）")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_traditional_fwi.yaml"),
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

    # --- 目标速度场 ---
    tgt_cfg = cfg.get("target") or {}
    model_path = _resolve_path(tgt_cfg["model_path"])
    sample_index = int(tgt_cfg.get("sample_index", 0))
    if not model_path.is_file():
        raise SystemExit(f"Data not found: {model_path}")
    data = np.load(model_path)  # (N, 1, 70, 70)
    target_vel_np = data[sample_index, 0].astype(np.float32)  # (70, 70) m/s

    # --- 目标波场（正演一次，无梯度）---
    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )  # (5, 1000, 70)

    # --- 可视化参数 ---
    viz = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    # --- 输出目录 ---
    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"traditional_fwi_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "model_path": str(model_path),
        "output_dir": str(out_dir),
        "device_used": str(device),
        "config_path": str(cfg_path.resolve()),
        "manifold_root": str(_MANIFOLD_ROOT),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    # --- 优化参数 ---
    fwi_cfg = cfg.get("fwi") or {}
    opt_steps = int(fwi_cfg.get("iterations", 300))
    lr = float(fwi_cfg.get("lr", 0.02))
    snapshots = int(fwi_cfg.get("snapshots", 10))
    loss_type = str(fwi_cfg.get("loss_type", "mse"))
    reg_type = str(fwi_cfg.get("reg_type", "none"))
    reg_lambda = float(fwi_cfg.get("reg_lambda", 0.0))
    use_cosine = bool(fwi_cfg.get("cosine_annealing", True))
    eta_min = float(fwi_cfg.get("eta_min", 0.0))

    # --- 初始化速度场 ---
    init_cfg = cfg.get("init") or {}
    init_method = str(init_cfg.get("method", "smooth"))
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))
    v_init = build_init_velocity(target_vel_np, init_method, device, smooth_sigma)
    v_init_np = v_denormalize_np(v_init)

    # =========================================================================
    # Pre-check: wave residual diagnostic for specified velocity models
    # =========================================================================
    pc_cfg = cfg.get("pre_check") or {}
    if pc_cfg.get("enabled", False):
        pc_shots = list(pc_cfg.get("shots", [wave_plot_shot]))
        pc_models = list(pc_cfg.get("models") or [])

        def _precheck_wave_errors(vel_np: np.ndarray) -> dict:
            with torch.no_grad():
                v_phys = torch.from_numpy(vel_np).float().to(device).clamp(1500.0, 4500.0)
                pred_w = seismic_master_forward_modeling(v_phys)
                residual = pred_w - target_wave
                l2  = float(F.mse_loss(pred_w, target_wave).item())
                l1  = float(F.l1_loss(pred_w, target_wave).item())
                rel = float((residual.norm() / (target_wave.norm() + 1e-12)).item())
            return {"pred_wave": pred_w, "residual": residual, "l2": l2, "l1": l1, "rel": rel}

        pc_results: list[dict] = []
        for entry in pc_models:
            label = str(entry.get("label", "?"))
            if label == "init":
                vel_np = v_init_np
            elif label == "target":
                vel_np = target_vel_np
            else:
                p = _resolve_path(entry["path"])
                idx = int(entry.get("index", 0))
                vel_np = np.load(p)[idx, 0].astype(np.float32)
            err = _precheck_wave_errors(vel_np)
            err["label"] = label
            err["vel_np"] = vel_np
            pc_results.append(err)
            mae_pc, ssim_pc = velocity_mae_ssim(vel_np, target_vel_np, device, vel_vmin, vel_vmax)
            print(
                f"[pre_check] {label:<16}  "
                f"wave_l2={err['l2']:.6f}  wave_l1={err['l1']:.6f}  rel={err['rel']:.4f}  "
                f"vel_MAE={mae_pc:.1f} m/s  vel_SSIM={ssim_pc:.4f}"
            )

        # figure: for each shot, show target / pred / residual per model
        for shot in pc_shots:
            tw_shot = target_wave[shot].detach().cpu().numpy()
            n_models = len(pc_results)
            nrows = n_models + 1   # +1 for target row
            ncols = 3
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows))
            axes = np.atleast_2d(axes).reshape(nrows, ncols)

            wlim_all = float(np.abs(tw_shot).max()) + 1e-6
            for r in pc_results:
                pw_shot = r["pred_wave"][shot].detach().cpu().numpy()
                wlim_all = max(wlim_all, float(np.abs(pw_shot).max()))

            # row 0: target wavefield
            im = axes[0, 0].imshow(tw_shot.T, cmap="seismic", aspect="auto",
                                   vmin=-wlim_all, vmax=wlim_all)
            axes[0, 0].set_title(f"target wave  shot {shot}", fontsize=8)
            axes[0, 0].axis("off"); plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
            axes[0, 1].axis("off"); axes[0, 2].axis("off")

            for i, r in enumerate(pc_results):
                row = i + 1
                pw_shot  = r["pred_wave"][shot].detach().cpu().numpy()
                res_shot = r["residual"][shot].detach().cpu().numpy()
                rlim = float(np.abs(res_shot).max()) + 1e-6

                im0 = axes[row, 0].imshow(pw_shot.T, cmap="seismic", aspect="auto",
                                          vmin=-wlim_all, vmax=wlim_all)
                axes[row, 0].set_title(f"{r['label']}  pred", fontsize=8)
                axes[row, 0].axis("off"); plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

                im1 = axes[row, 1].imshow(res_shot.T, cmap="seismic", aspect="auto",
                                          vmin=-rlim, vmax=rlim)
                axes[row, 1].set_title(
                    f"{r['label']}  residual\nL2={r['l2']:.4f}  L1={r['l1']:.4f}  rel={r['rel']:.4f}",
                    fontsize=7,
                )
                axes[row, 1].axis("off"); plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

                im2 = axes[row, 2].imshow(r["vel_np"], cmap="viridis", aspect="auto",
                                          vmin=vel_vmin, vmax=vel_vmax)
                axes[row, 2].set_title(f"{r['label']}  velocity", fontsize=8)
                axes[row, 2].axis("off"); plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

            plt.suptitle(f"Pre-check: wave residuals  |  shot {shot}  |  model60[{sample_index}]",
                         fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f"pre_check_shot{shot}.png", dpi=200, bbox_inches="tight")
            plt.close()

        # free GPU tensors from pre_check before optimization
        for r in pc_results:
            del r["pred_wave"], r["residual"]

    # =========================================================================

    v = v_init.clone().requires_grad_(True)

    opt = torch.optim.Adam([v], lr=lr)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=eta_min)
    snap_times = np.unique(np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int))
    snap_times = np.clip(snap_times, 0, opt_steps)

    # --- 历史记录 ---
    hist_vel:    list[np.ndarray] = []
    hist_grad:   list[np.ndarray] = []
    hist_labels: list[str] = []

    def capture(label: str) -> None:
        hist_vel.append(v_denormalize_np(v))
        hist_labels.append(label)
        grad_np = (
            v.grad.detach().cpu().numpy().copy()
            if v.grad is not None
            else np.zeros((70, 70), dtype=np.float32)
        )
        hist_grad.append(grad_np)

    losses:     list[float] = []
    wave_losses: list[float] = []
    reg_losses:  list[float] = []
    mae_hist:    list[float] = []
    ssim_hist:   list[float] = []
    grad_hist:   list[float] = []

    if 0 in snap_times:
        capture("iter 0 (init)")

    # --- 优化循环 ---
    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)

        v_phys = v_denormalize_tensor(v).clamp(1500.0, 4500.0)
        pred_wave = seismic_master_forward_modeling(v_phys)

        if loss_type == "l1":
            wave_loss = F.l1_loss(pred_wave, target_wave)
        else:  # default: mse
            wave_loss = F.mse_loss(pred_wave, target_wave)

        use_reg = reg_lambda > 0.0 and reg_type != "none"
        if use_reg:
            if reg_type == "tv":
                reg = tv_loss(v)
            else:  # tikhonov
                reg = tikhonov_loss(v)
            loss = wave_loss + reg_lambda * reg
            reg_val = (reg_lambda * reg).item()
        else:
            loss = wave_loss
            reg_val = 0.0

        loss.backward()
        grad_total = v.grad.norm().item() if v.grad is not None else 0.0
        opt.step()
        if use_cosine:
            scheduler.step()

        # 将速度约束在归一化范围内
        v.data.clamp_(-1.0, 1.0)

        losses.append(loss.item())
        wave_losses.append(wave_loss.item())
        reg_losses.append(reg_val)
        grad_hist.append(grad_total)

        with torch.no_grad():
            pv = v_denormalize_np(v)
            mae, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            capture(f"iter {it + 1}")

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            if use_reg:
                print(
                    f"iter {it+1}/{opt_steps}  "
                    f"total={loss.item():.6f}  wave={wave_loss.item():.6f}  reg={reg_val:.6f}  "
                    f"MAE={mae:.1f} m/s  SSIM={ssim_v:.4f}  ||∇v||={grad_total:.4e}"
                )
            else:
                print(
                    f"iter {it+1}/{opt_steps}  wave={loss.item():.6f}  "
                    f"MAE={mae:.1f} m/s  SSIM={ssim_v:.4f}  ||∇v||={grad_total:.4e}"
                )

    # --- 图1：优化演化快照（速度场 + 误差 + ∇v）---
    ncols = 1 + len(hist_vel)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 7.5))
    axes = np.atleast_2d(axes).reshape(3, ncols)

    # 第 0 列：目标 + 初始误差
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("target v\n(model60)", fontsize=8)
    axes[0, 0].axis("off")
    err0 = v_init_np - target_vel_np
    err0_lim = max(float(np.abs(err0).max()), 1e-6)
    axes[1, 0].imshow(err0, cmap="coolwarm", aspect="auto", vmin=-err0_lim, vmax=err0_lim)
    axes[1, 0].set_title("init error", fontsize=8)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")   # 第 0 列梯度格留空（init 时无梯度）

    for j, (vel, gj, lbl) in enumerate(zip(hist_vel, hist_grad, hist_labels)):
        col = j + 1
        axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(lbl, fontsize=8)
        axes[0, col].axis("off")
        err = vel - target_vel_np
        elim = max(float(np.abs(err).max()), 1e-6)
        axes[1, col].imshow(err, cmap="coolwarm", aspect="auto", vmin=-elim, vmax=elim)
        axes[1, col].axis("off")
        glim = max(float(np.percentile(np.abs(gj), 99)), 1e-8)
        axes[2, col].imshow(gj, cmap="coolwarm", aspect="auto", vmin=-glim, vmax=glim)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("error (m/s)", fontsize=9)
    axes[2, 0].set_ylabel("∇v (FWI grad)", fontsize=9)
    plt.suptitle(
        f"Traditional FWI | loss={loss_type}  reg={reg_type}(λ={reg_lambda}) | "
        f"model60[{sample_index}]  init={init_method}(σ={smooth_sigma})",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图2：最终结果（速度 + 波场对比）---
    with torch.no_grad():
        pred_vel_np = v_denormalize_np(v)
        v_phys_final = v_denormalize_tensor(v).clamp(1500.0, 4500.0)
        pred_wave_final = seismic_master_forward_modeling(v_phys_final)

    err_v = pred_vel_np - target_vel_np
    err_lim_v = float(np.abs(err_v).max()) + 1e-6
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
    reg_str = f"  reg={reg_losses[-1]:.6f}" if reg_lambda > 0.0 else ""
    plt.suptitle(
        f"Final | wave={wave_losses[-1]:.6f}{reg_str}  total={losses[-1]:.6f}  "
        f"vel MAE={mae_hist[-1]:.1f} m/s  SSIM={ssim_hist[-1]:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 图3：指标曲线 ---
    iters = list(range(1, len(losses) + 1))
    use_reg_plot = reg_lambda > 0.0 and reg_type != "none"
    if use_reg_plot:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        axes[0, 0].plot(iters, wave_losses, color="C0")
        axes[0, 0].set_ylabel(f"wave loss ({loss_type})")
        axes[0, 0].set_title("Data loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(iters, reg_losses, color="C5")
        axes[0, 1].set_ylabel(f"λ×reg  (λ={reg_lambda})")
        axes[0, 1].set_title(f"Regularization ({reg_type})")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(iters, losses, color="C0", label="total")
        axes[1, 0].plot(iters, wave_losses, color="C0", linestyle="--", alpha=0.5, label="wave")
        axes[1, 0].plot(iters, reg_losses, color="C5", linestyle="--", alpha=0.5, label="reg")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Total loss")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(iters, mae_hist, color="C1")
        axes[1, 1].set_ylabel("MAE (m/s)")
        axes[1, 1].set_title("Velocity MAE")
        axes[1, 1].grid(True, alpha=0.3)
        axes[2, 0].plot(iters, ssim_hist, color="C2")
        axes[2, 0].set_ylabel("SSIM")
        axes[2, 0].set_xlabel("Iteration")
        axes[2, 0].set_title("Velocity SSIM")
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 1].semilogy(iters, grad_hist, color="C3")
        axes[2, 1].set_ylabel("||∇v|| (log)")
        axes[2, 1].set_xlabel("Iteration")
        axes[2, 1].set_title("Gradient norm of v")
        axes[2, 1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
        axes[0, 0].plot(iters, losses, color="C0")
        axes[0, 0].set_ylabel(f"wave loss ({loss_type})")
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
        axes[1, 1].semilogy(iters, grad_hist, color="C3")
        axes[1, 1].set_ylabel("||∇v|| (log scale)")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_title("Gradient norm of v")
        axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle("Optimization metrics", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "final_total_loss": float(losses[-1]),
        "final_wave_loss": float(wave_losses[-1]),
        "final_reg_loss": float(reg_losses[-1]),
        "final_vel_mae_m_s": float(mae_hist[-1]),
        "final_vel_ssim": float(ssim_hist[-1]),
        "reg_type": reg_type,
        "reg_lambda": reg_lambda,
        "init_method": init_method,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
