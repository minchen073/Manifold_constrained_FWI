#!/usr/bin/env python3
"""
DDIM Latent Smoothing Optimization — CurveVel-B

优化 z 使得 DDIM(z) 经高斯光滑后接近真解的高斯光滑版本：
  min_z  MSE( gaussian_smooth(DDIM(z) * 1500 + 3000),
              gaussian_smooth(v_target) )

高斯光滑通过 F.conv2d 实现，全程可微，梯度流经 smooth → DDIM → z。
评估指标：生成速度场（未光滑）与真解之间的 MAE 和 SSIM。

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/Curvevel_B/ddim_smooth_opt/demo_ddim_smooth_opt.py \\
    --config exp/Curvevel_B/ddim_smooth_opt/config_ddim_smooth_opt.yaml
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

_SCRIPT_DIR    = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]   # .../Manifold_constrained_FWI
_TRAINING_DIR  = _MANIFOLD_ROOT / "training"
_CURVE_VEL_B   = _MANIFOLD_ROOT.parent / "CurveVelB"

for _p in [str(_CURVE_VEL_B), str(_TRAINING_DIR), str(_MANIFOLD_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


# ---------------------------------------------------------------------------
# config / path helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# velocity helpers
# ---------------------------------------------------------------------------

def v_denormalize_np(v: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().numpy()
    return v.astype(np.float32).squeeze() * 1500.0 + 3000.0


def v_denormalize_tensor(v: torch.Tensor) -> torch.Tensor:
    return v * 1500.0 + 3000.0


def vel_to_ssim_tensor(vel_np: np.ndarray, vmin: float, vmax: float) -> torch.Tensor:
    arr = np.squeeze(vel_np).astype(np.float32)
    t = torch.from_numpy(arr).view(1, 1, arr.shape[-2], arr.shape[-1])
    return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0)


def velocity_mae_ssim(
    pred_np: np.ndarray,
    target_np: np.ndarray,
    device: torch.device,
    vmin: float,
    vmax: float,
) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred_np - target_np)))
    t1 = vel_to_ssim_tensor(pred_np,  vmin, vmax).to(device)
    t2 = vel_to_ssim_tensor(target_np, vmin, vmax).to(device)
    ssim_val = float(pytorch_ssim.ssim(t1, t2, window_size=11, size_average=True).item())
    return mae, ssim_val


# ---------------------------------------------------------------------------
# differentiable Gaussian smoothing
# ---------------------------------------------------------------------------

def make_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    ks = max(3, int(6 * sigma + 1) | 1)   # odd, at least 3
    r  = ks // 2
    x  = torch.arange(-r, r + 1, dtype=torch.float32)
    g  = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
    g  = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, ks, ks).to(device)
    return kernel


def gaussian_smooth(v: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """v: any shape with last two dims (H, W) → (1,1,H,W), differentiable."""
    h, w = v.shape[-2], v.shape[-1]
    pad  = kernel.shape[-1] // 2
    return F.conv2d(v.reshape(1, 1, h, w), kernel, padding=pad)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: Path, training_yaml: Path | None):
    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}")

    torch_dtype = torch.float32
    if training_yaml is not None and training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            tcfg = yaml.safe_load(f)
        s = str((tcfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        torch_dtype = torch.float16 if s in ("float16", "fp16") else torch.float32

    wrapper  = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)
    ddpm_sch = DDPMScheduler.from_pretrained(str(checkpoint))
    ddim_sch = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
    return wrapper, ddpm_sch, ddim_sch


# ---------------------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------------------

def sample_with_grad(
    wrapper, ddim_sched, z: torch.Tensor, num_steps: int, eta: float = 0.0
) -> torch.Tensor:
    ddim_sched.set_timesteps(num_steps)
    x = z
    for t in ddim_sched.timesteps:
        model_output = wrapper(x, t).sample
        x = ddim_sched.step(
            model_output, t, x, eta=eta, use_clipped_model_output=False
        ).prev_sample
    return x


def sample_no_grad(
    wrapper, ddim_sched, z: torch.Tensor, num_steps: int, eta: float = 0.0
) -> torch.Tensor:
    with torch.no_grad():
        return sample_with_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)


# ---------------------------------------------------------------------------
# scheduler helper
# ---------------------------------------------------------------------------

def make_scheduler(opt, sched_type: str, t_max: int, eta_min: float):
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM 潜变量高斯平滑优化")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_smooth_opt.yaml"),
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

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- DDIM model ---
    ddim_cfg     = cfg.get("ddim") or {}
    ckpt         = _resolve_path(ddim_cfg["checkpoint"])
    num_steps    = int(ddim_cfg.get("num_steps", 3))
    eta          = float(ddim_cfg.get("eta", 0.0))

    training_yaml = _TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml"
    if not training_yaml.is_file():
        training_yaml = ckpt.parent / "config_used.yaml"

    wrapper, _, ddim_sch = load_model(ckpt, training_yaml)
    wrapper = wrapper.to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # --- target velocity ---
    cv_cfg       = cfg.get("curvevel") or {}
    model_path   = _resolve_path(cv_cfg["model60_path"])
    sample_index = int(cv_cfg.get("sample_index", 0))
    if not model_path.is_file():
        raise SystemExit(f"Data not found: {model_path}")
    target_vel_np = np.load(model_path)[sample_index, 0].astype(np.float32)  # (70,70) m/s

    # --- smoothing ---
    sm_cfg     = cfg.get("smoothing") or {}
    sigma      = float(sm_cfg.get("sigma", 10.0))
    gauss_kern = make_gaussian_kernel(sigma, device)

    # 与 ddim_joint_opt 一致：先归一化到 [-1,1]，再平滑
    target_vel_norm = torch.from_numpy(
        (target_vel_np / 1500.0 - 2.0).astype(np.float32)
    ).to(device)  # (70,70)
    target_smooth = gaussian_smooth(target_vel_norm, gauss_kern).detach()  # (1,1,70,70) 归一化空间

    # --- visualization params ---
    viz      = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))

    # --- output dir ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_smooth_opt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "checkpoint": str(ckpt), "output_dir": str(out_dir),
        "device_used": str(device),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    # --- optimization config ---
    opt_cfg     = cfg.get("optimization") or {}
    total_steps = int(opt_cfg.get("total_steps", 300))
    lr          = float(opt_cfg.get("lr", 0.02))
    snapshots   = int(opt_cfg.get("snapshots", 8))
    sched_cfg   = opt_cfg.get("scheduler") or {}
    sched_type  = str(sched_cfg.get("type", "cosine")).lower()
    sched_eta   = float(sched_cfg.get("eta_min", 0.0))

    print(
        f"[DDIM Smooth Opt | CurveVel-B]  sample={sample_index}  "
        f"sigma={sigma}  total={total_steps}  lr={lr}  "
        f"num_steps={num_steps}  device={device}"
    )

    # --- initialize z ---
    z = torch.randn(1, 1, 70, 70, dtype=torch.float32, device=device, requires_grad=True)

    opt   = torch.optim.Adam([z], lr=lr)
    sched = make_scheduler(opt, sched_type, total_steps, sched_eta)

    snap_times = set(
        int(x) for x in np.clip(
            np.round(np.linspace(0, total_steps, snapshots, endpoint=True)).astype(int),
            0, total_steps,
        )
    )

    # --- history ---
    hist_vel:    list[np.ndarray] = []
    hist_vel_sm: list[np.ndarray] = []   # smooth(v_gen) in physical m/s at each snapshot
    hist_labels: list[str]        = []
    snap_mae:    list[float]      = []
    snap_ssim:   list[float]      = []

    losses:    list[float] = []
    mae_hist:  list[float] = []
    ssim_hist: list[float] = []
    grad_hist: list[float] = []

    # snapshot at iter 0
    if 0 in snap_times:
        with torch.no_grad():
            v0_t  = sample_no_grad(wrapper, ddim_sch, z, num_steps, eta)
            v0    = v_denormalize_np(v0_t)
            v0_sm = gaussian_smooth(v0_t, gauss_kern).squeeze().cpu().numpy() * 1500.0 + 3000.0
        m0, s0 = velocity_mae_ssim(v0, target_vel_np, device, vel_vmin, vel_vmax)
        hist_vel.append(v0); hist_vel_sm.append(v0_sm)
        hist_labels.append("iter 0 (init)")
        snap_mae.append(m0); snap_ssim.append(s0)

    # --- optimization loop ---
    for it in range(total_steps):
        opt.zero_grad(set_to_none=True)

        v_gen    = sample_with_grad(wrapper, ddim_sch, z, num_steps, eta)  # (1,1,70,70) ∈ [-1,1]
        v_gen_sm = gaussian_smooth(v_gen, gauss_kern)                       # 归一化空间平滑
        loss     = F.mse_loss(v_gen_sm, target_smooth)

        loss.backward()
        grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
        opt.step()
        if sched is not None:
            sched.step()

        losses.append(loss.item())
        grad_hist.append(grad_norm)

        with torch.no_grad():
            v_np = v_denormalize_np(v_gen.detach())
            mae, ssim_v = velocity_mae_ssim(v_np, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            sm_snap = v_gen_sm.detach().squeeze().cpu().numpy() * 1500.0 + 3000.0
            hist_vel.append(v_np.copy()); hist_vel_sm.append(sm_snap)
            hist_labels.append(f"iter {it + 1}")
            snap_mae.append(mae); snap_ssim.append(ssim_v)

        if (it + 1) % max(1, total_steps // 8) == 0 or it == 0:
            print(
                f"iter {it+1:4d}/{total_steps}  "
                f"loss={loss.item():.6f}  MAE={mae:.1f} m/s  SSIM={ssim_v:.4f}  "
                f"||∇z||={grad_norm:.4e}"
            )

    # =========================================================================
    # Figure 1: velocity snapshots  (3 rows: gen | smooth(gen) | smooth error)
    # =========================================================================
    sm_np = target_smooth.squeeze().detach().cpu().numpy() * 1500.0 + 3000.0  # physical m/s

    ncols = 1 + len(hist_vel)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 8.0))
    axes = np.atleast_2d(axes).reshape(3, ncols)

    # col 0: target (row 0) and smooth(target) (row 1), row 2 empty
    im = axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto",
                           vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("target v", fontsize=8); axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[1, 0].imshow(sm_np, cmap="viridis", aspect="auto",
                           vmin=vel_vmin, vmax=vel_vmax)
    axes[1, 0].set_title("smooth(target)", fontsize=8); axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[2, 0].axis("off")

    for j, (vel, vel_sm, lbl, mae_s, ssim_s) in enumerate(
        zip(hist_vel, hist_vel_sm, hist_labels, snap_mae, snap_ssim)
    ):
        col = j + 1

        # row 0: raw generated velocity
        im = axes[0, col].imshow(vel, cmap="viridis", aspect="auto",
                                 vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(f"{lbl}\nMAE={mae_s:.0f}  SSIM={ssim_s:.3f}", fontsize=7)
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

        # row 1: smooth(gen) in physical units
        im = axes[1, col].imshow(vel_sm, cmap="viridis", aspect="auto",
                                 vmin=vel_vmin, vmax=vel_vmax)
        axes[1, col].set_title("smooth(gen)", fontsize=7)
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)

        # row 2: smooth residual = smooth(gen) - smooth(target)
        sm_err = vel_sm - sm_np
        sm_elim = max(float(np.abs(sm_err).max()), 1e-6)
        im = axes[2, col].imshow(sm_err, cmap="coolwarm", aspect="auto",
                                 vmin=-sm_elim, vmax=sm_elim)
        axes[2, col].set_title("smooth residual", fontsize=7)
        axes[2, col].axis("off")
        plt.colorbar(im, ax=axes[2, col], fraction=0.046)

    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("smooth(v) (m/s)", fontsize=9)
    axes[2, 0].set_ylabel("smooth residual", fontsize=9)
    plt.suptitle(
        f"DDIM Smooth Opt | CurveVel-B sample {sample_index}  σ={sigma}  "
        f"lr={lr}  steps={total_steps}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "snapshots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 2: final comparison  (target | generated | error | smooth comparison)
    # =========================================================================
    with torch.no_grad():
        v_final    = sample_no_grad(wrapper, ddim_sch, z, num_steps, eta)  # (1,1,70,70) 归一化
        v_final_np = v_denormalize_np(v_final)                              # 物理 m/s
        # smooth 在归一化空间算，再反归一化到物理单位用于可视化
        v_final_sm = (
            gaussian_smooth(v_final, gauss_kern).squeeze().cpu().numpy() * 1500.0 + 3000.0
        )

    err_final = v_final_np - target_vel_np
    err_lim   = max(float(np.abs(err_final).max()), 1e-6)
    sm_err    = v_final_sm - sm_np
    sm_err_lim = max(float(np.abs(sm_err).max()), 1e-6)
    mae_f, ssim_f = velocity_mae_ssim(v_final_np, target_vel_np, device, vel_vmin, vel_vmax)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (target_vel_np, "target v (m/s)",       "viridis",  vel_vmin,    vel_vmax),
        (v_final_np,    "generated v (m/s)",     "viridis",  vel_vmin,    vel_vmax),
        (err_final,     "v error (m/s)",          "coolwarm", -err_lim,    err_lim),
        (sm_np,         "smooth(target) (m/s)",  "viridis",  vel_vmin,    vel_vmax),
        (v_final_sm,    "smooth(gen) (m/s)",     "viridis",  vel_vmin,    vel_vmax),
        (sm_err,        "smooth error (m/s)",     "coolwarm", -sm_err_lim, sm_err_lim),
    ]
    for ax, (arr, title, cmap, vl, vh) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vl, vmax=vh)
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        f"Final | MAE={mae_f:.1f} m/s  SSIM={ssim_f:.4f}  "
        f"smooth_loss={losses[-1]:.6f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 3: metrics curves
    # =========================================================================
    iters = list(range(1, len(losses) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    axes[0, 0].plot(iters, losses, color="C0")
    axes[0, 0].set_title("Smooth MSE loss"); axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, mae_hist, color="C1")
    axes[0, 1].set_title("Velocity MAE (raw gen vs target)")
    axes[0, 1].set_ylabel("MAE (m/s)"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, ssim_hist, color="C2")
    axes[1, 0].set_title("Velocity SSIM (raw gen vs target)")
    axes[1, 0].set_ylabel("SSIM"); axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(iters, grad_hist, color="C3")
    axes[1, 1].set_title("Gradient norm ||∇z||")
    axes[1, 1].set_ylabel("||∇z|| (log)"); axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"DDIM Smooth Opt | sample {sample_index}  σ={sigma}  "
        f"final MAE={mae_f:.1f} m/s  SSIM={ssim_f:.4f}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # summary
    # =========================================================================
    summary = {
        "sample_index":    sample_index,
        "sigma":           sigma,
        "total_steps":     total_steps,
        "lr":              lr,
        "final_smooth_loss": float(losses[-1]),
        "final_vel_mae_m_s": mae_f,
        "final_vel_ssim":    ssim_f,
        "out_dir":           str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. MAE={mae_f:.1f} m/s  SSIM={ssim_f:.4f}  Output: {out_dir}")


if __name__ == "__main__":
    main()
