#!/usr/bin/env python3
"""
Alternating FWI + SSI.

Repeating cycle:
  ┌─ FWI phase ──────────────────────────────────────────────────────────┐
  │  Optimize μ directly with wave L1/MSE loss (Adam + CosineAnnealing)  │
  │  μ ← μ − lr·∇_μ L(forward(μ), wave_target),  clamp(μ, −1, 1)       │
  └──────────────────────────────────────────────────────────────────────┘
       ↓  after fwi_steps iterations
  ┌─ SSI phase ──────────────────────────────────────────────────────────┐
  │  1. Jump:      x_{t_j} = √ᾱ_{t_j}·μ + √(1−ᾱ_{t_j})·ε             │
  │  2. Inversion: z_inv = ddim_inversion(x_{t_j})   [ascending ODE]    │
  │  3. Generation: μ_new = ddim_sample(z_inv)        [descending ODE]  │
  │  μ ← μ_new  (new leaf tensor, detached)                             │
  └──────────────────────────────────────────────────────────────────────┘
       ↓  repeat

Run (from Manifold_constrained_FWI/):
  uv run python exp/DDIM/ddim_fwi_ssi/demo_ddim_fwi_ssi.py \\
    --config exp/DDIM/ddim_fwi_ssi/config_ddim_fwi_ssi.yaml
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

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter

_SCRIPT_DIR    = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]
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


def v_denormalize_tensor(v: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → physical velocity m/s, keeps gradient."""
    return v * 1500.0 + 3000.0


def v_denormalize_np(v: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().numpy()
    return v.astype(np.float32).squeeze() * 1500.0 + 3000.0


def v_normalize_np(vel_np: np.ndarray) -> np.ndarray:
    return (vel_np / 1500.0 - 2.0).astype(np.float32)


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
    t1  = vel_to_ssim_tensor(pred_np,  vmin, vmax).to(device)
    t2  = vel_to_ssim_tensor(target_np, vmin, vmax).to(device)
    ssim_val = float(pytorch_ssim.ssim(t1, t2, window_size=11, size_average=True).item())
    return mae, ssim_val


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_ddim_pipeline(checkpoint: Path, training_yaml: Path | None):
    """Returns (wrapper, inv_sched) where inv_sched is a DDIMScheduler."""
    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}")

    torch_dtype = torch.float32
    if training_yaml is not None and training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s = str((cfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        torch_dtype = torch.float16 if s in ("float16", "fp16") else torch.float32

    wrapper    = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)
    ddpm_sched = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    inv_sched  = DDIMScheduler.from_config(ddpm_sched.config, clip_sample=False)
    return wrapper, inv_sched


# ---------------------------------------------------------------------------
# forward wave operator
# ---------------------------------------------------------------------------

def forward_wave(mu: torch.Tensor) -> torch.Tensor:
    """μ in [-1,1], shape (1,1,70,70) → wave (5,1000,70). Gradient flows."""
    v_phys = v_denormalize_tensor(mu.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


# ---------------------------------------------------------------------------
# DDIM inversion (ascending ODE, no grad)
# ---------------------------------------------------------------------------

def ddim_inversion(
    wrapper,
    inv_sched,
    x_tj: torch.Tensor,
    jump_step_idx: int,
) -> torch.Tensor:
    """Run DDIM inversion from x_{t_j} to x_{t_max} (ascending ODE, η=0).

    inv_sched must have set_timesteps(inv_steps) called before use.

    Args:
        x_tj:          noisy sample at timestep t_j, shape (1,1,70,70)
        jump_step_idx: index into ascending timestep list
                       inversion_ts = inv_sched.timesteps.flip(0)
                       x_tj lives at inversion_ts[jump_step_idx]

    Returns:
        z_inv: shape (1,1,70,70), approximately N(0,I)
    """
    inversion_ts = inv_sched.timesteps.flip(0)  # ascending: [t_small, ..., t_large]
    n_inv = len(inversion_ts)

    image = x_tj.clone()
    with torch.no_grad():
        for i in range(jump_step_idx, n_inv - 1):
            t_cur  = inversion_ts[i]
            t_next = inversion_ts[i + 1]

            alpha_t      = inv_sched.alphas_cumprod[t_cur].to(image.device)
            alpha_t_next = inv_sched.alphas_cumprod[t_next].to(image.device)

            eps_theta = wrapper(image, t_cur).sample
            x0_pred   = (image - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()
            image     = alpha_t_next.sqrt() * x0_pred + (1 - alpha_t_next).sqrt() * eps_theta

    return image  # (1, 1, 70, 70)


# ---------------------------------------------------------------------------
# DDIM sampling (descending ODE, no grad)
# ---------------------------------------------------------------------------

def ddim_sample(
    wrapper,
    sample_sched,
    z: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Standard DDIM sampling (η=0). z: (1,1,70,70) → (1,1,70,70).

    Calls set_timesteps internally — use a dedicated sample_sched instance
    separate from inv_sched to avoid state interference.
    """
    sample_sched.set_timesteps(num_steps)
    image = z
    with torch.no_grad():
        for t in sample_sched.timesteps:
            model_output = wrapper(image, t).sample
            image = sample_sched.step(
                model_output, t, image, eta=0.0, use_clipped_model_output=False
            ).prev_sample
    return image  # (1, 1, 70, 70)


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------

def build_init(
    init_type: str,
    target_vel_np: np.ndarray,
    smooth_sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Build initial velocity model μ_0 as (1,1,70,70) requires_grad tensor.

    Matches official RED-DiffEq prepare_initial_model():
      smoothed    — normalize first, then gaussian_filter
      homogeneous — fill with min of top row of normalized field (near-surface velocity)
      linear      — linspace from global min to max along depth axis
      zero        — constant 0 in normalized space (= 3000 m/s uniform)
    """
    v_np = v_normalize_np(target_vel_np)   # (70,70), normalize first

    if init_type == "smoothed":
        mu_np = gaussian_filter(v_np, sigma=smooth_sigma).astype(np.float32)
    elif init_type == "homogeneous":
        min_top_row = float(np.min(v_np[0, :]))
        mu_np = np.full(v_np.shape, min_top_row, dtype=np.float32)
    elif init_type == "linear":
        v_min, v_max = float(np.min(v_np)), float(np.max(v_np))
        depth_grad = np.linspace(v_min, v_max, v_np.shape[0], dtype=np.float32)
        mu_np = np.tile(depth_grad.reshape(-1, 1), (1, v_np.shape[1]))
    else:  # "zero"
        mu_np = np.zeros(v_np.shape, dtype=np.float32)

    mu = torch.from_numpy(mu_np).reshape(1, 1, 70, 70).to(device)
    return mu.clamp(-1.0, 1.0).requires_grad_(True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Alternating FWI + SSI")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_fwi_ssi.yaml"),
    )
    args     = parser.parse_args()
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

    wrapper, inv_sched = load_ddim_pipeline(ckpt, training_yaml)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)

    inv_steps    = int(dcfg.get("inv_steps",    100))
    sample_steps = int(dcfg.get("sample_steps",  20))

    # Two independent schedulers — ddim_sample calls set_timesteps on sample_sched,
    # which must NOT be the same object as inv_sched.
    sample_sched = DDIMScheduler.from_config(inv_sched.config, clip_sample=False)

    # Set up inversion timestep grid (fixed for all cycles)
    inv_sched.set_timesteps(inv_steps)
    inversion_ts = inv_sched.timesteps.flip(0)  # ascending: [t_small, ..., t_large]

    # --- SSI jump parameters ---
    ssi_cfg       = cfg.get("ssi") or {}
    jump_step_idx = int(ssi_cfg.get("jump_step_idx", 5))
    random_eps    = bool(ssi_cfg.get("random_eps", True))
    eps_seed      = int(ssi_cfg.get("eps_seed", 0))

    jump_step_idx = min(jump_step_idx, inv_steps - 2)
    t_j           = int(inversion_ts[jump_step_idx].item())
    alpha_bar_tj  = inv_sched.alphas_cumprod[t_j].to(device)

    # --- data ---
    cv           = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    sample_index = int(cv.get("sample_index", 0))
    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    target_vel_np = np.load(model60_path)[sample_index, 0].astype(np.float32)  # (70,70) m/s

    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # --- FWI config ---
    fwi_cfg    = cfg.get("fwi") or {}
    num_cycles = int(fwi_cfg.get("num_cycles", 5))
    fwi_steps  = int(fwi_cfg.get("fwi_steps", 60))
    lr         = float(fwi_cfg.get("lr", 0.03))
    lr_min     = float(fwi_cfg.get("lr_min", 0.0))
    wave_loss  = str(fwi_cfg.get("wave_loss", "l1")).lower()

    # --- visualization config ---
    viz            = cfg.get("visualization") or {}
    vel_vmin       = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax       = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    # --- output directory ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_fwi_ssi_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "checkpoint": str(ckpt), "model60_path": str(model60_path),
        "output_dir": str(out_dir), "device_used": str(device),
        "t_j": t_j, "alpha_bar_tj": float(alpha_bar_tj.item()),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    # --- initialization ---
    init_cfg     = cfg.get("initialization") or {}
    init_type    = str(init_cfg.get("type", "zero"))
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))

    mu = build_init(init_type, target_vel_np, smooth_sigma, device)

    print(
        f"[FWI+SSI]  checkpoint={ckpt.name}  init={init_type}  "
        f"num_cycles={num_cycles}  fwi_steps={fwi_steps}  lr={lr}\n"
        f"  inv_steps={inv_steps}  sample_steps={sample_steps}  "
        f"jump_step_idx={jump_step_idx}/{inv_steps-1}  t_j={t_j}  "
        f"ᾱ_tj={alpha_bar_tj.item():.4f}  random_eps={random_eps}  device={device}"
    )

    # =========================================================================
    # Tracking structures
    # =========================================================================
    # Per-step metrics (concatenated across all FWI steps in all cycles)
    wave_loss_hist: list[float] = []
    mae_hist:       list[float] = []
    ssim_hist:      list[float] = []
    # Boundaries: global step index at which each SSI is applied
    ssi_boundaries: list[int]   = []

    # Snapshots: initial + after each FWI phase + after each SSI phase
    snap_vels:   list[np.ndarray] = []
    snap_labels: list[str]        = []

    # Capture initial state
    snap_vels.append(v_denormalize_np(mu).copy())
    snap_labels.append("init")

    # =========================================================================
    # Main alternating loop
    # =========================================================================
    global_step = 0

    for cycle in range(num_cycles):
        print(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")

        # ------------------------------------------------------------------ #
        # FWI phase                                                           #
        # ------------------------------------------------------------------ #
        # Re-create optimizer each cycle (mu is replaced after SSI)
        optimizer = torch.optim.Adam([mu], lr=lr)
        lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=fwi_steps, eta_min=lr_min
        )

        for step in range(fwi_steps):
            optimizer.zero_grad(set_to_none=True)

            wave_pred = forward_wave(mu)
            if wave_loss == "l1":
                loss = F.l1_loss(wave_pred, target_wave)
            else:
                loss = F.mse_loss(wave_pred, target_wave)

            loss.backward()
            optimizer.step()
            lr_sched.step()

            with torch.no_grad():
                mu.data.clamp_(-1.0, 1.0)

            with torch.no_grad():
                vel_np = v_denormalize_np(mu)
                mae, ssim_val = velocity_mae_ssim(vel_np, target_vel_np, device, vel_vmin, vel_vmax)

            wave_loss_hist.append(loss.item())
            mae_hist.append(mae)
            ssim_hist.append(ssim_val)
            global_step += 1

            if (step + 1) % max(1, fwi_steps // 4) == 0:
                print(
                    f"  FWI {step+1:3d}/{fwi_steps}  "
                    f"loss={loss.item():.6f}  MAE={mae:.1f} m/s  SSIM={ssim_val:.4f}  "
                    f"lr={lr_sched.get_last_lr()[0]:.5f}"
                )

        # Snapshot after FWI
        snap_vels.append(v_denormalize_np(mu).copy())
        snap_labels.append(f"C{cycle+1} FWI")

        # ------------------------------------------------------------------ #
        # SSI phase                                                           #
        # ------------------------------------------------------------------ #
        eps_cycle_seed = (eps_seed + cycle) if random_eps else eps_seed
        g = torch.Generator(device=device).manual_seed(eps_cycle_seed)
        eps = torch.randn(1, 1, 70, 70, device=device, generator=g)

        with torch.no_grad():
            mu_det = mu.detach()
            x_tj = alpha_bar_tj.sqrt() * mu_det + (1 - alpha_bar_tj).sqrt() * eps

        z_inv  = ddim_inversion(wrapper, inv_sched, x_tj, jump_step_idx)
        mu_new = ddim_sample(wrapper, sample_sched, z_inv, sample_steps)

        # Replace μ with new leaf tensor (detached from DDIM graph)
        mu = mu_new.detach().clamp(-1.0, 1.0).requires_grad_(True)

        ssi_boundaries.append(global_step)

        z_np   = z_inv.detach().cpu().numpy().squeeze()
        zmean, zstd = float(z_np.mean()), float(z_np.std())
        ssi_vel_np = v_denormalize_np(mu)
        mae_ssi, ssim_ssi = velocity_mae_ssim(ssi_vel_np, target_vel_np, device, vel_vmin, vel_vmax)
        print(
            f"  SSI  z: mean={zmean:+.3f} std={zstd:.3f}  "
            f"→ MAE={mae_ssi:.1f} m/s  SSIM={ssim_ssi:.4f}"
        )

        # Snapshot after SSI
        snap_vels.append(ssi_vel_np.copy())
        snap_labels.append(f"C{cycle+1} SSI")

    # =========================================================================
    # Final evaluation
    # =========================================================================
    final_vel_np = v_denormalize_np(mu)
    mae_final, ssim_final = velocity_mae_ssim(final_vel_np, target_vel_np, device, vel_vmin, vel_vmax)
    print(f"\nFinal  MAE={mae_final:.1f} m/s  SSIM={ssim_final:.4f}")

    with torch.no_grad():
        wave_final = forward_wave(mu)

    # =========================================================================
    # Figure 1: velocity evolution (snapshots)
    # =========================================================================
    ncols = len(snap_vels) + 1  # +1 for ground truth
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.5))
    axes = np.array(axes).ravel()

    im = axes[0].imshow(target_vel_np, cmap="viridis", aspect="equal", vmin=vel_vmin, vmax=vel_vmax)
    axes[0].set_title("target", fontsize=8)
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    for j, (vel, lbl) in enumerate(zip(snap_vels, snap_labels)):
        ax = axes[j + 1]
        ax.imshow(vel, cmap="viridis", aspect="equal", vmin=vel_vmin, vmax=vel_vmax)
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")

    plt.suptitle(
        f"FWI+SSI evolution  |  init={init_type}  "
        f"num_cycles={num_cycles}  fwi_steps={fwi_steps}  "
        f"t_j={t_j}  model60[{sample_index}]",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 2: final result
    # =========================================================================
    sh         = wave_plot_shot
    tw         = target_wave[sh].detach().cpu().numpy()
    pw         = wave_final[sh].detach().cpu().numpy()
    ew         = pw - tw
    err_v      = final_vel_np - target_vel_np
    err_lim_v  = max(float(np.abs(err_v).max()), 1.0)
    wlim       = max(float(np.abs(tw).max()), float(np.abs(pw).max())) + 1e-6
    elw        = float(np.abs(ew).max()) + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np, "target v (m/s)",  "viridis",  vel_vmin,    vel_vmax),
        (final_vel_np,  "pred v (m/s)",    "viridis",  vel_vmin,    vel_vmax),
        (err_v,         "v error (m/s)",   "coolwarm", -err_lim_v,  err_lim_v),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="equal", vmin=vm0, vmax=vm1)
        ax.set_title(title); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[1], [
        (tw.T, f"target wave shot {sh}", "seismic", -wlim, wlim),
        (pw.T, f"pred wave shot {sh}",   "seismic", -wlim, wlim),
        (ew.T, "wave residual",          "seismic", -elw,  elw),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(
        f"FWI+SSI  |  {init_type} init  cycles={num_cycles}  fwi_steps={fwi_steps}  "
        f"t_j={t_j}  MAE={mae_final:.1f} m/s  SSIM={ssim_final:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 3: metrics over all FWI steps (SSI boundaries as vertical lines)
    # =========================================================================
    total_fwi_steps = len(wave_loss_hist)
    iters = list(range(1, total_fwi_steps + 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(iters, wave_loss_hist, color="C0", lw=1.0)
    axes[0].set_title("Wave loss"); axes[0].set_ylabel(wave_loss.upper())
    axes[0].set_xlabel("FWI step (cumulative)"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, mae_hist, color="C1", lw=1.0)
    axes[1].set_title("Velocity MAE"); axes[1].set_ylabel("MAE (m/s)")
    axes[1].set_xlabel("FWI step (cumulative)"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, ssim_hist, color="C2", lw=1.0)
    axes[2].set_title("Velocity SSIM"); axes[2].set_ylabel("SSIM")
    axes[2].set_xlabel("FWI step (cumulative)"); axes[2].grid(True, alpha=0.3)

    # Mark SSI boundaries
    for ax in axes:
        for b in ssi_boundaries:
            ax.axvline(b, color="red", lw=1.0, ls="--", alpha=0.6)
    axes[0].legend(["wave loss", "SSI"], loc="upper right", fontsize=8)

    plt.suptitle(
        f"FWI+SSI metrics  |  {num_cycles} cycles × {fwi_steps} FWI steps  "
        f"t_j={t_j}  model60[{sample_index}]\n"
        f"(red dashed: SSI projection)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Summary JSON
    # =========================================================================
    summary = {
        "init_type":        init_type,
        "num_cycles":       num_cycles,
        "fwi_steps":        fwi_steps,
        "lr":               lr,
        "t_j":              t_j,
        "alpha_bar_tj":     float(alpha_bar_tj.item()),
        "jump_step_idx":    jump_step_idx,
        "sample_index":     sample_index,
        "final_wave_loss":  float(wave_loss_hist[-1]) if wave_loss_hist else None,
        "final_vel_mae_m_s": float(mae_final),
        "final_vel_ssim":   float(ssim_final),
        "out_dir":          str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {out_dir}")
    print(f"  MAE={mae_final:.1f} m/s  SSIM={ssim_final:.4f}")


if __name__ == "__main__":
    main()
