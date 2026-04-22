#!/usr/bin/env python3
"""
DDIM Wave-MSE Optimization with Inversion (DDIM-OPT-INV-WAVE)

Three-phase pipeline using the DDPM/DDIM model:

  Phase 1 — Optimize DDIM latent z with wave MSE:
      z ~ N(0,I),  Adam optimizer
      For each step:
          x̂ = ddim_sample_grad(z, sample_steps)   [no torch.no_grad → gradients flow]
          L = MSE(forward(x̂), wave_target)
          z ← z − lr·∇_z L

  Phase 2 — DDIM Inversion of x̂₁ (Phase-1 final result):
      x_{t_j} = √ᾱ_{t_j}·x̂₁ + √(1-ᾱ_{t_j})·ε,   ε ~ N(0,I)
      z_inv   = ddim_inversion(x_{t_j})              [ascending ODE, no_grad]

  Phase 3 — Continue optimization from z_inv:
      Same loop as Phase 1, initializing z = z_inv.detach()

Outputs (per run):
  phase1_evolution.png     — optimization snapshots (Phase 1)
  phase1_result.png        — velocity & wave comparison after Phase 1
  inversion_noise.png      — inverted noise z_inv visualization
  phase3_evolution.png     — optimization snapshots (Phase 3)
  phase3_result.png        — velocity & wave comparison after Phase 3
  metrics_combined.png     — MAE / SSIM / loss curves for both phases
  summary.json

Run (from Manifold_constrained_FWI/):
  uv run python exp/DDIM/ddim_opt_inv_wave/demo_ddim_opt_inv_wave.py \\
    --config exp/DDIM/ddim_opt_inv_wave/config_ddim_opt_inv_wave.yaml
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

_SCRIPT_DIR    = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]  # exp/DDIM/ddim_opt_inv_wave → Manifold_constrained_FWI
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
from src.seismic import seismic_master_forward_modeling


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def v_denormalize_tensor(v_norm: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → physical velocity in m/s, keeps gradient."""
    return v_norm * 1500.0 + 3000.0


def v_denormalize_np(v_norm: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(v_norm, torch.Tensor):
        v_norm = v_norm.detach().float().cpu().numpy()
    return v_norm.astype(np.float32).squeeze() * 1500.0 + 3000.0


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
    """Load OpenFWIUNetWrapper + two separate DDIMSchedulers (inv / sample)."""
    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}")

    torch_dtype = torch.float32
    if training_yaml is not None and training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s = str((cfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        torch_dtype = torch.float16 if s in ("float16", "fp16") else torch.float32

    wrapper  = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)
    ddpm_sch = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    # Two independent schedulers so set_timesteps calls never interfere
    inv_sched    = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
    sample_sched = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
    return wrapper, inv_sched, sample_sched


# ---------------------------------------------------------------------------
# DDIM sampling — differentiable (for optimization)
# ---------------------------------------------------------------------------

def ddim_sample_grad(
    wrapper, sched, z: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """DDIM sampling (η=0) with full gradient flow through z.

    Implements the DDIM step manually so gradients are clean:
        ε_θ    = model(x_t, t)
        x̂_0  = (x_t − √(1−ᾱ_t)·ε_θ) / √(ᾱ_t)
        x_{t'} = √(ᾱ_{t'})·x̂_0 + √(1−ᾱ_{t'})·ε_θ
    """
    sched.set_timesteps(num_steps)
    ts    = sched.timesteps          # descending: [T, ..., 0]
    image = z                        # (1, 1, 70, 70)
    for i, t in enumerate(ts):
        alpha_t = sched.alphas_cumprod[t.item()].to(image.device)
        prev_t  = ts[i + 1].item() if i + 1 < len(ts) else -1
        alpha_t_prev = (
            sched.alphas_cumprod[prev_t].to(image.device)
            if prev_t >= 0
            else torch.ones(1, device=image.device)
        )
        eps_theta = wrapper(image, t).sample
        x0_pred   = (image - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()
        image     = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * eps_theta
    return image  # (1, 1, 70, 70)


def ddim_sample_nograd(
    wrapper, sched, z: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """DDIM sampling without gradient (for visualization / evaluation)."""
    with torch.no_grad():
        return ddim_sample_grad(wrapper, sched, z.detach(), num_steps)


# ---------------------------------------------------------------------------
# DDIM inversion — ascending ODE (no_grad)
# ---------------------------------------------------------------------------

def ddim_inversion(
    wrapper, inv_sched, x_tj: torch.Tensor, jump_step_idx: int
) -> torch.Tensor:
    """Run DDIM inversion from x_{t_j} → x_{t_max} (ascending ODE, η=0).

    Args:
        x_tj:          Noisy sample at timestep t_j, shape (1,1,70,70).
        jump_step_idx: Index into ascending timestep list
                       (inversion_ts = inv_sched.timesteps.flip(0)).

    Returns:
        x_{t_max} ≈ N(0, I), shape (1,1,70,70).
    """
    inversion_ts = inv_sched.timesteps.flip(0)   # ascending: [t_small, ..., t_max]
    n_inv  = len(inversion_ts)
    image  = x_tj.clone()
    with torch.no_grad():
        for i in range(jump_step_idx, n_inv - 1):
            t_cur  = inversion_ts[i]
            t_next = inversion_ts[i + 1]
            alpha_t      = inv_sched.alphas_cumprod[t_cur].to(image.device)
            alpha_t_next = inv_sched.alphas_cumprod[t_next].to(image.device)
            eps_theta = wrapper(image, t_cur).sample
            x0_pred   = (image - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()
            image     = alpha_t_next.sqrt() * x0_pred + (1 - alpha_t_next).sqrt() * eps_theta
    return image


# ---------------------------------------------------------------------------
# wave forward
# ---------------------------------------------------------------------------

def forward_wave(pred_norm: torch.Tensor) -> torch.Tensor:
    """DDIM output [-1,1] (70×70) → physical velocity → wave (5,1000,70)."""
    v_phys = v_denormalize_tensor(pred_norm.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


# ---------------------------------------------------------------------------
# optimization loop (shared by Phase 1 and Phase 3)
# ---------------------------------------------------------------------------

def run_optimization(
    wrapper,
    sample_sched,
    sample_steps: int,
    z_init: torch.Tensor,
    target_wave: torch.Tensor,
    target_vel_np: np.ndarray,
    opt_steps: int,
    lr: float,
    reg_weight: float,
    snapshots: int,
    vel_vmin: float,
    vel_vmax: float,
    phase_label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Run wave-MSE optimization for `opt_steps` iterations.

    Returns:
        z_final:      optimized latent, shape (1,1,70,70)
        x_hat_final:  decoded velocity in [-1,1], shape (1,1,70,70)
        histories:    dict with loss/MAE/SSIM/grad histories and snapshot lists
    """
    device = z_init.device
    z = z_init.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([z], lr=lr)

    d        = z.numel()
    r_star_sq = float(d - 1)
    reg_min   = 0.5 * r_star_sq * (1.0 - math.log(r_star_sq))

    snap_times = set(
        int(x) for x in np.clip(
            np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int),
            0, opt_steps,
        )
    )

    # snapshot containers
    hist_vel:      list[np.ndarray] = []
    hist_z:        list[np.ndarray] = []
    hist_grad:     list[np.ndarray] = []
    hist_grad_vel: list[np.ndarray] = []
    hist_labels:   list[str]        = []

    losses, wave_losses, reg_losses = [], [], []
    mae_hist, ssim_hist, grad_hist  = [], [], []

    def capture(label: str, grad_vel_np: np.ndarray | None = None) -> None:
        x_snap = ddim_sample_nograd(wrapper, sample_sched, z, sample_steps)
        hist_vel.append(v_denormalize_np(x_snap).copy())
        hist_z.append(z.detach().cpu().numpy().squeeze().copy())
        hist_labels.append(label)
        hist_grad.append(
            z.grad.detach().cpu().numpy().squeeze().copy()
            if z.grad is not None
            else np.zeros(z.shape[-2:], dtype=np.float32)
        )
        hist_grad_vel.append(
            grad_vel_np if grad_vel_np is not None
            else np.zeros(z.shape[-2:], dtype=np.float32)
        )

    if 0 in snap_times:
        capture(f"{phase_label} iter 0 (init)")

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        x_hat     = ddim_sample_grad(wrapper, sample_sched, z, sample_steps)
        x_hat.retain_grad()
        wave_pred = forward_wave(x_hat)
        wave_mse  = F.mse_loss(wave_pred, target_wave)

        if reg_weight > 0.0:
            z_norm  = torch.norm(z.view(-1), p=2)
            reg     = -(d - 1) * torch.log(z_norm + 1e-8) + z_norm ** 2 / 2 - reg_min
            loss    = wave_mse + reg_weight * reg
            loss.backward()
            with torch.no_grad():
                r2           = z_norm.item() ** 2 + 1e-16
                grad_reg_vec = reg_weight * z * (1.0 - (d - 1) / r2)
            reg_val = (reg_weight * reg).item()
        else:
            loss = wave_mse
            loss.backward()
            reg_val = 0.0

        opt.step()

        losses.append(loss.item())
        wave_losses.append(wave_mse.item())
        reg_losses.append(reg_val)
        grad_hist.append(z.grad.norm().item() if z.grad is not None else 0.0)

        with torch.no_grad():
            pv = v_denormalize_np(x_hat)
            mae_v, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae_v)
        ssim_hist.append(ssim_v)

        if (it + 1) in snap_times:
            gv = x_hat.grad.detach().cpu().numpy().squeeze().copy() if x_hat.grad is not None else None
            capture(f"{phase_label} iter {it + 1}", grad_vel_np=gv)

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            print(
                f"  [{phase_label}] iter {it+1}/{opt_steps}  "
                f"wave MSE={wave_mse.item():.6f}  reg={reg_val:.6f}  "
                f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  "
                f"||∇z||={grad_hist[-1]:.4e}"
            )

    with torch.no_grad():
        x_hat_final = ddim_sample_nograd(wrapper, sample_sched, z, sample_steps)

    histories = dict(
        losses=losses, wave_losses=wave_losses, reg_losses=reg_losses,
        mae_hist=mae_hist, ssim_hist=ssim_hist, grad_hist=grad_hist,
        hist_vel=hist_vel, hist_z=hist_z, hist_grad=hist_grad,
        hist_grad_vel=hist_grad_vel, hist_labels=hist_labels,
    )
    return z.detach(), x_hat_final, histories


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

def plot_evolution(
    target_vel_np: np.ndarray,
    z_init_np: np.ndarray,
    hist: dict,
    vel_vmin: float,
    vel_vmax: float,
    title: str,
    out_path: Path,
) -> None:
    """4-row evolution grid: velocity / z / ∇z / ∇x̂."""
    hist_vel      = hist["hist_vel"]
    hist_z        = hist["hist_z"]
    hist_grad     = hist["hist_grad"]
    hist_grad_vel = hist["hist_grad_vel"]
    hist_labels   = hist["hist_labels"]

    ncols  = 1 + len(hist_vel)
    z0lim  = max(3.0, float(np.abs(z_init_np).max()) + 0.1,
                 max((float(np.abs(hz).max()) for hz in hist_z), default=0.0) + 0.1)

    fig, axes = plt.subplots(4, ncols, figsize=(2.6 * ncols, 10.0))
    axes = np.atleast_2d(axes).reshape(4, ncols)

    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("target v\n(model60)", fontsize=8)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(z_init_np, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
    axes[1, 0].set_title("init z", fontsize=8)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[3, 0].axis("off")
    axes[0, 0].set_ylabel("v (m/s)", fontsize=9)
    axes[1, 0].set_ylabel("noise z", fontsize=9)
    axes[2, 0].set_ylabel("∇z", fontsize=9)
    axes[3, 0].set_ylabel("∇x̂₀", fontsize=9)

    for j, (vel, zj, gj, gvj, lbl) in enumerate(
        zip(hist_vel, hist_z, hist_grad, hist_grad_vel, hist_labels)
    ):
        col = j + 1
        axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(lbl, fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-z0lim, vmax=z0lim)
        axes[1, col].axis("off")
        glim = max(float(np.percentile(np.abs(gj), 99)), 1e-8)
        axes[2, col].imshow(gj, cmap="coolwarm", aspect="auto", vmin=-glim, vmax=glim)
        axes[2, col].axis("off")
        gvlim = max(float(np.percentile(np.abs(gvj), 99)), 1e-8)
        axes[3, col].imshow(gvj, cmap="coolwarm", aspect="auto", vmin=-gvlim, vmax=gvlim)
        axes[3, col].axis("off")

    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_result(
    target_vel_np: np.ndarray,
    pred_vel_np: np.ndarray,
    target_wave: torch.Tensor,
    pred_wave: torch.Tensor,
    vel_vmin: float,
    vel_vmax: float,
    wave_plot_shot: int,
    title: str,
    out_path: Path,
) -> None:
    """2×3 result: velocity row + wave row."""
    err_v     = pred_vel_np - target_vel_np
    err_lim_v = float(np.max(np.abs(err_v))) + 1e-6
    sh = wave_plot_shot
    tw = target_wave[sh].detach().cpu().numpy()
    pw = pred_wave[sh].detach().cpu().numpy()
    ew = pw - tw
    wlim = max(np.abs(tw).max(), np.abs(pw).max()) + 1e-6
    elw  = float(np.abs(ew).max()) + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (arr, ttl, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np,    "target v (m/s)",  "viridis",  vel_vmin,   vel_vmax),
        (pred_vel_np,      "pred v (m/s)",    "viridis",  vel_vmin,   vel_vmax),
        (err_v,            "v error (m/s)",   "coolwarm", -err_lim_v, err_lim_v),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(ttl)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, ttl, cmap, vm0, vm1) in zip(axes[1], [
        (tw.T, f"target wave shot {sh}", "seismic", -wlim,  wlim),
        (pw.T, f"pred wave shot {sh}",   "seismic", -wlim,  wlim),
        (ew.T, "wave residual",          "seismic", -elw,   elw),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_combined_metrics(
    hist1: dict,
    hist3: dict,
    phase1_steps: int,
    phase3_steps: int,
    out_path: Path,
) -> None:
    """Side-by-side MAE / SSIM / wave-loss / ||∇z|| for Phase 1 and Phase 3."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    iters1 = list(range(1, phase1_steps + 1))
    iters3 = list(range(phase1_steps + 1, phase1_steps + phase3_steps + 1))

    for ax, key, ylabel, title, log in [
        (axes[0, 0], "wave_losses", "wave MSE",     "Wave-space loss",    False),
        (axes[0, 1], "mae_hist",    "MAE (m/s)",    "Velocity MAE",       False),
        (axes[1, 0], "ssim_hist",   "SSIM",         "Velocity SSIM",      False),
        (axes[1, 1], "grad_hist",   "||∇z|| (log)", "Gradient norm of z", True),
    ]:
        fn = ax.semilogy if log else ax.plot
        fn(iters1, hist1[key], color="C0", label="Phase 1")
        fn(iters3, hist3[key], color="C1", label="Phase 3")
        ax.axvline(phase1_steps + 0.5, color="gray", linestyle="--", linewidth=0.8, label="inversion")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Iteration")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phase 1 → DDIM Inversion → Phase 3", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM opt + inversion + opt")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_opt_inv_wave.yaml"),
    )
    args    = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path.resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device = torch.device(cfg.get("device", "cuda:0"))
    cp.cuda.Device(device.index if device.index is not None else 0).use()

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- DDIM pipeline ---
    dcfg = cfg.get("ddim") or {}
    ckpt = _resolve_path(dcfg["checkpoint"])
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    training_yaml_arg = dcfg.get("training_yaml")
    if training_yaml_arg:
        training_yaml = _resolve_path(training_yaml_arg)
    else:
        candidate = (ckpt.parent / "config_used.yaml").resolve()
        training_yaml = candidate if candidate.is_file() else (
            _TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml"
        ).resolve()

    wrapper, inv_sched, sample_sched = load_ddim_pipeline(ckpt, training_yaml)
    wrapper = wrapper.to(device).eval()

    sample_steps = int(dcfg.get("sample_steps", 20))
    inv_steps    = int(dcfg.get("inv_steps", 100))

    # Set up inversion schedule (fixed; never changed again)
    inv_sched.set_timesteps(inv_steps)
    inversion_ts = inv_sched.timesteps.flip(0)  # ascending

    # --- data ---
    cv = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    sample_index = int(cv.get("sample_index", 0))
    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    raw = np.load(model60_path)[sample_index, 0].astype(np.float32)   # (70,70) m/s
    target_vel_np = raw
    x0_norm = torch.from_numpy(
        (raw / 1500.0 - 2.0).astype(np.float32)
    ).reshape(1, 1, 70, 70).to(device)

    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # --- inversion parameters ---
    inv_cfg       = cfg.get("inversion") or {}
    jump_step_idx = int(inv_cfg.get("jump_step_idx", 5))
    jump_step_idx = min(jump_step_idx, inv_steps - 2)
    eps_seed      = int(inv_cfg.get("eps_seed", 0))
    t_j           = inversion_ts[jump_step_idx].item()
    alpha_bar_tj  = inv_sched.alphas_cumprod[t_j].to(device)

    # --- optimization parameters ---
    opt_cfg      = cfg.get("optimization") or {}
    seed_z       = int(opt_cfg.get("seed_z", 42))
    phase1_steps = int(opt_cfg.get("phase1_steps", 300))
    phase3_steps = int(opt_cfg.get("phase3_steps", 300))
    lr           = float(opt_cfg.get("lr", 0.05))
    snapshots    = int(opt_cfg.get("snapshots", 8))
    reg_weight   = float(opt_cfg.get("reg_weight", 0.0))

    # --- visualization ---
    viz           = cfg.get("visualization") or {}
    vel_vmin      = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax      = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    # --- output directory ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_opt_inv_wave_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "checkpoint": str(ckpt), "model60_path": str(model60_path),
        "output_dir": str(out_dir), "device_used": str(device),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    print(
        f"[DDIM-OPT-INV-WAVE]  checkpoint={ckpt.name}  "
        f"sample_steps={sample_steps}  inv_steps={inv_steps}  "
        f"jump_step_idx={jump_step_idx}/{inv_steps-1}  t_j={t_j}  "
        f"alpha_bar={alpha_bar_tj.item():.4f}  device={device}"
    )
    print(
        f"  sqrt(ᾱ_tj)={alpha_bar_tj.sqrt().item():.4f}  "
        f"sqrt(1-ᾱ_tj)={(1-alpha_bar_tj).sqrt().item():.4f}"
    )

    # =========================================================================
    # Phase 1: optimize z with wave MSE
    # =========================================================================
    print(f"\n=== Phase 1: optimize z ({phase1_steps} iters) ===")
    g_z   = torch.Generator(device=device).manual_seed(seed_z)
    z_p1  = torch.randn(1, 1, 70, 70, device=device, dtype=torch.float32, generator=g_z)

    z1_final, x_hat1, hist1 = run_optimization(
        wrapper=wrapper,
        sample_sched=sample_sched,
        sample_steps=sample_steps,
        z_init=z_p1,
        target_wave=target_wave,
        target_vel_np=target_vel_np,
        opt_steps=phase1_steps,
        lr=lr,
        reg_weight=reg_weight,
        snapshots=snapshots,
        vel_vmin=vel_vmin,
        vel_vmax=vel_vmax,
        phase_label="P1",
    )
    vel1_np = v_denormalize_np(x_hat1)
    mae1, ssim1 = velocity_mae_ssim(vel1_np, target_vel_np, device, vel_vmin, vel_vmax)
    print(f"  Phase 1 done  MAE={mae1:.1f} m/s  SSIM={ssim1:.4f}")

    # =========================================================================
    # Phase 2: DDIM inversion of x_hat1
    # =========================================================================
    print(f"\n=== Phase 2: DDIM inversion (jump_step_idx={jump_step_idx}, t_j={t_j}) ===")
    g_eps  = torch.Generator(device=device).manual_seed(eps_seed)
    eps    = torch.randn(1, 1, 70, 70, device=device, generator=g_eps)
    x_tj   = alpha_bar_tj.sqrt() * x_hat1.detach() + (1 - alpha_bar_tj).sqrt() * eps
    z_inv  = ddim_inversion(wrapper, inv_sched, x_tj, jump_step_idx)

    z_inv_np = z_inv.cpu().numpy().squeeze()
    print(
        f"  z_inv  mean={z_inv_np.mean():+.4f}  std={z_inv_np.std():.4f}  "
        f"(ideal: mean≈0, std≈1)"
    )

    # =========================================================================
    # Phase 3: continue optimization from z_inv
    # =========================================================================
    print(f"\n=== Phase 3: optimize z_inv ({phase3_steps} iters) ===")
    z3_final, x_hat3, hist3 = run_optimization(
        wrapper=wrapper,
        sample_sched=sample_sched,
        sample_steps=sample_steps,
        z_init=z_inv,
        target_wave=target_wave,
        target_vel_np=target_vel_np,
        opt_steps=phase3_steps,
        lr=lr,
        reg_weight=reg_weight,
        snapshots=snapshots,
        vel_vmin=vel_vmin,
        vel_vmax=vel_vmax,
        phase_label="P3",
    )
    vel3_np = v_denormalize_np(x_hat3)
    mae3, ssim3 = velocity_mae_ssim(vel3_np, target_vel_np, device, vel_vmin, vel_vmax)
    print(f"  Phase 3 done  MAE={mae3:.1f} m/s  SSIM={ssim3:.4f}")

    # =========================================================================
    # Figures
    # =========================================================================

    # --- Phase 1 evolution ---
    plot_evolution(
        target_vel_np, z_p1.cpu().numpy().squeeze(), hist1,
        vel_vmin, vel_vmax,
        title=f"Phase 1 evolution  |  sample_steps={sample_steps}  model60[{sample_index}]",
        out_path=out_dir / "phase1_evolution.png",
    )

    # --- Phase 1 final result ---
    with torch.no_grad():
        wave1 = forward_wave(x_hat1)
    plot_result(
        target_vel_np, vel1_np, target_wave, wave1,
        vel_vmin, vel_vmax, wave_plot_shot,
        title=f"Phase 1 result  |  MAE={mae1:.1f} m/s  SSIM={ssim1:.4f}  wave MSE={hist1['wave_losses'][-1]:.6f}",
        out_path=out_dir / "phase1_result.png",
    )

    # --- inverted noise z_inv ---
    zlim = max(3.0, float(np.percentile(np.abs(z_inv_np), 99)) + 0.1)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(z_inv_np, cmap="coolwarm", aspect="auto", vmin=-zlim, vmax=zlim)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(
        f"Inverted noise z_inv\n"
        f"t_j={t_j}  (step {jump_step_idx}/{inv_steps-1})  "
        f"ᾱ_tj={alpha_bar_tj.item():.4f}\n"
        f"mean={z_inv_np.mean():+.4f}  std={z_inv_np.std():.4f}",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "inversion_noise.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Phase 3 evolution ---
    plot_evolution(
        target_vel_np, z_inv_np, hist3,
        vel_vmin, vel_vmax,
        title=f"Phase 3 evolution  |  starting from z_inv  |  sample_steps={sample_steps}",
        out_path=out_dir / "phase3_evolution.png",
    )

    # --- Phase 3 final result ---
    with torch.no_grad():
        wave3 = forward_wave(x_hat3)
    plot_result(
        target_vel_np, vel3_np, target_wave, wave3,
        vel_vmin, vel_vmax, wave_plot_shot,
        title=f"Phase 3 result  |  MAE={mae3:.1f} m/s  SSIM={ssim3:.4f}  wave MSE={hist3['wave_losses'][-1]:.6f}",
        out_path=out_dir / "phase3_result.png",
    )

    # --- combined metrics ---
    plot_combined_metrics(hist1, hist3, phase1_steps, phase3_steps,
                          out_path=out_dir / "metrics_combined.png")

    # --- summary ---
    summary = {
        "phase1": {
            "final_wave_mse": float(hist1["wave_losses"][-1]),
            "final_vel_mae_m_s": mae1,
            "final_vel_ssim": ssim1,
        },
        "inversion": {
            "t_j": t_j,
            "jump_step_idx": jump_step_idx,
            "inv_steps": inv_steps,
            "alpha_bar_tj": float(alpha_bar_tj.item()),
            "z_inv_mean": float(z_inv_np.mean()),
            "z_inv_std": float(z_inv_np.std()),
        },
        "phase3": {
            "final_wave_mse": float(hist3["wave_losses"][-1]),
            "final_vel_mae_m_s": mae3,
            "final_vel_ssim": ssim3,
        },
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {out_dir}")
    print(f"  Phase 1: MAE={mae1:.1f} m/s  SSIM={ssim1:.4f}")
    print(f"  Phase 3: MAE={mae3:.1f} m/s  SSIM={ssim3:.4f}")


if __name__ == "__main__":
    main()
