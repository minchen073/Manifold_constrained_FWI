#!/usr/bin/env python3
"""
RED-DiffEq: Regularization by Denoising with Diffusion Models for FWI.

Reproduces the RED-DiffEq algorithm using our pretrained DDPM/DDIM model
as the diffusion prior and seismic_master_forward_modeling as the forward op.

Algorithm (per optimization step):
  1. wave_pred = forward(μ)                            [differentiable]
  2. L_obs     = L1(wave_pred, wave_target)
  3. x_t       = √ᾱ_t·μ + √(1-ᾱ_t)·ε,  t ~ U(0, max_t)
  4. ε_pred    = model(x_t, t)                         [no grad through model]
  5. L_reg     = mean((ε_pred − ε).detach() ⊙ μ)
  6. L         = L_obs + λ·L_reg
  7. μ ← Adam(∇_μ L),  then clamp(μ, −1, 1)

Key difference from ddim_opt_inv_wave:
  - Optimize μ (velocity) directly, NOT the DDIM latent z.
  - Diffusion model is used purely as a RED regularizer (no sampling in main loop).

Run (from Manifold_constrained_FWI/):
  uv run python exp/DDIM/ddim_red_diffeq/demo_ddim_red_diffeq.py \\
    --config exp/DDIM/ddim_red_diffeq/config_ddim_red_diffeq.yaml
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
from scipy.ndimage import gaussian_filter

_SCRIPT_DIR    = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]  # exp/DDIM/ddim_red_diffeq → root
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
    from diffusers import DDPMScheduler, DDIMScheduler
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
    """Physical velocity m/s → [-1, 1]."""
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


def load_model(checkpoint: Path, training_yaml: Path | None):
    """Load OpenFWI wrapper + DDPMScheduler (for alphas_cumprod)."""
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
    return wrapper, ddpm_sch


# ---------------------------------------------------------------------------
# forward wave operator
# ---------------------------------------------------------------------------

def forward_wave(mu: torch.Tensor) -> torch.Tensor:
    """μ in [-1,1], shape (1,1,70,70) → wave (5,1000,70), gradient flows."""
    v_phys = v_denormalize_tensor(mu.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


# ---------------------------------------------------------------------------
# RED regularization
# ---------------------------------------------------------------------------

def red_reg_loss(
    wrapper,
    alphas_cumprod: torch.Tensor,
    mu: torch.Tensor,
    max_timestep: int,
    sigma_x0: float,
    use_time_weight: bool,
    rng: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """RED-DiffEq regularization loss.

    L_reg = mean((ε_pred − ε).detach() ⊙ μ)

    The gradient ∂L_reg/∂μ = (ε_pred − ε) / N acts as the score signal
    that pulls μ toward the diffusion model's learned data manifold.

    Args:
        mu:  current velocity estimate, shape (1,1,70,70), requires_grad=True
        rng: per-step Generator for reproducibility

    Returns:
        (reg_loss scalar, timestep used)
    """
    # Optional pre-noise
    if sigma_x0 > 0:
        noise_x0 = torch.randn(mu.shape, device=mu.device, generator=rng)
        x0_pred  = mu + sigma_x0 * noise_x0
    else:
        x0_pred = mu

    # Random timestep
    t_val = int(torch.randint(0, max_timestep, (1,), generator=rng, device=mu.device).item())
    t_tensor = torch.tensor([t_val], device=mu.device, dtype=torch.long)

    # Forward diffusion: x_t = √ᾱ_t · x0_pred + √(1-ᾱ_t) · ε
    alpha_bar = alphas_cumprod[t_val].to(mu.device)
    eps       = torch.randn(x0_pred.shape, device=mu.device, generator=rng)
    x_t       = alpha_bar.sqrt() * x0_pred + (1 - alpha_bar).sqrt() * eps

    # Noise prediction: no grad through model weights
    # x_t.detach() so backward graph for mu is not extended through the model
    with torch.no_grad():
        eps_pred = wrapper(x_t.detach(), t_tensor).sample   # (1,1,70,70)

        # Prior velocity estimate x̂₀ (linear combination of x_t and ε_pred):
        #   x̂₀ = (x_t − √(1−ᾱ_t)·ε_pred) / √ᾱ_t
        x0_hat = (x_t.detach() - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt().clamp(min=1e-8)

    # Gradient field (detached): ≈ score function
    gradient_field = (eps_pred - eps).detach()

    # Optional time-weighting: w_t = √((1-ᾱ_t)/ᾱ_t)
    if use_time_weight:
        w_t = ((1 - alpha_bar) / alpha_bar.clamp(min=1e-8)).sqrt()
        gradient_field = gradient_field * w_t

    # RED loss: inner product of gradient field and x0_pred
    # Gradient flows through x0_pred (→ mu) only
    loss = (gradient_field * x0_pred).mean()

    return loss, t_val, x0_hat  # x0_hat: (1,1,70,70), detached


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

    Matches the official RED-DiffEq prepare_initial_model() implementation:
      smoothed    — normalize first, then gaussian_filter (linear ops commute, but kept consistent)
      homogeneous — fill with min value of the top row of the normalized field (near-surface velocity)
      linear      — linspace from global min to max along the depth axis
      zero        — constant 0 in normalized space (= 3000 m/s uniform)
    """
    v_np = v_normalize_np(target_vel_np)   # (70,70), normalized first

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
    mu = mu.clamp(-1.0, 1.0).requires_grad_(True)
    return mu


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RED-DiffEq FWI")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_red_diffeq.yaml"),
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

    # --- model ---
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

    wrapper, ddpm_sch = load_model(ckpt, training_yaml)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)   # never update model weights
    alphas_cumprod = ddpm_sch.alphas_cumprod.to(device)   # (1000,)

    # --- data ---
    cv = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    sample_index = int(cv.get("sample_index", 0))
    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    raw           = np.load(model60_path)[sample_index, 0].astype(np.float32)  # (70,70) m/s
    target_vel_np = raw

    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # --- initialization ---
    init_cfg     = cfg.get("initialization") or {}
    init_type    = str(init_cfg.get("type", "zero"))
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))

    mu = build_init(init_type, target_vel_np, smooth_sigma, device)
    mu_init_np = mu.detach().cpu().numpy().squeeze().copy()

    # --- optimization config ---
    opt_cfg    = cfg.get("optimization") or {}
    opt_steps  = int(opt_cfg.get("opt_steps", 300))
    lr         = float(opt_cfg.get("lr", 0.03))
    reg_lambda = float(opt_cfg.get("reg_lambda", 0.75))
    snapshots  = int(opt_cfg.get("snapshots", 8))
    wave_loss  = str(opt_cfg.get("wave_loss", "l1")).lower()

    red_cfg        = cfg.get("red") or {}
    max_timestep   = int(red_cfg.get("max_timestep", 1000))
    sigma_x0       = float(red_cfg.get("sigma_x0", 0.0001))
    use_time_weight = bool(red_cfg.get("use_time_weight", False))

    viz            = cfg.get("visualization") or {}
    vel_vmin       = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax       = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    # --- output directory ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"s{sample_index}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "checkpoint": str(ckpt), "model60_path": str(model60_path),
        "output_dir": str(out_dir), "device_used": str(device),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    print(
        f"[RED-DiffEq]  checkpoint={ckpt.name}  init={init_type}  "
        f"opt_steps={opt_steps}  lr={lr}  λ={reg_lambda}  "
        f"max_t={max_timestep}  σ_x0={sigma_x0}  device={device}"
    )

    # =========================================================================
    # Optimization loop
    # =========================================================================
    optimizer = torch.optim.Adam([mu], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_steps, eta_min=0.0)
    rng = torch.Generator(device=device).manual_seed(seed)

    snap_times = set(
        int(x) for x in np.clip(
            np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int),
            0, opt_steps,
        )
    )

    hist_vel:    list[np.ndarray] = []
    hist_mu:     list[np.ndarray] = []
    hist_x0hat:  list[np.ndarray] = []   # prior velocity field x̂₀ at snapshot step
    hist_t:      list[int]        = []   # timestep used at each snapshot
    hist_labels: list[str]        = []

    total_losses:   list[float] = []
    obs_losses:     list[float] = []
    reg_losses:     list[float] = []
    mae_hist:       list[float] = []
    ssim_hist:      list[float] = []
    t_hist:         list[int]   = []
    grad_obs_norms: list[float] = []
    grad_reg_norms: list[float] = []

    def capture(label: str, x0_hat: torch.Tensor | None, t_used: int) -> None:
        hist_vel.append(v_denormalize_np(mu).copy())
        hist_mu.append(mu.detach().cpu().numpy().squeeze().copy())
        hist_labels.append(label)
        hist_t.append(t_used)
        if x0_hat is not None:
            hist_x0hat.append(v_denormalize_np(x0_hat).copy())
        else:
            # iter-0 init: no RED step yet, show μ itself as placeholder
            hist_x0hat.append(v_denormalize_np(mu).copy())

    if 0 in snap_times:
        capture("iter 0 (init)", x0_hat=None, t_used=-1)

    for it in range(opt_steps):
        optimizer.zero_grad(set_to_none=True)

        # --- observation loss ---
        wave_pred = forward_wave(mu)
        if wave_loss == "l1":
            loss_obs = F.l1_loss(wave_pred, target_wave)
        else:
            loss_obs = F.mse_loss(wave_pred, target_wave)

        # --- RED regularization ---
        loss_reg, t_used, x0_hat = red_reg_loss(
            wrapper, alphas_cumprod, mu,
            max_timestep, sigma_x0, use_time_weight, rng,
        )

        grad_obs_v = torch.autograd.grad(loss_obs, mu, retain_graph=True)[0]
        grad_reg_v = torch.autograd.grad(loss_reg, mu, retain_graph=True)[0]
        grad_obs_norm_val = grad_obs_v.norm().item()
        grad_reg_norm_val = grad_reg_v.norm().item()

        loss = loss_obs + reg_lambda * loss_reg
        loss.backward()
        optimizer.step()
        scheduler.step()

        # clamp μ to valid normalized range
        with torch.no_grad():
            mu.data.clamp_(-1.0, 1.0)

        total_losses.append(loss.item())
        obs_losses.append(loss_obs.item())
        reg_losses.append(loss_reg.item())
        t_hist.append(t_used)
        grad_obs_norms.append(grad_obs_norm_val)
        grad_reg_norms.append(grad_reg_norm_val)

        with torch.no_grad():
            vel_np = v_denormalize_np(mu)
            mae, ssim_val = velocity_mae_ssim(vel_np, target_vel_np, device, vel_vmin, vel_vmax)
        mae_hist.append(mae)
        ssim_hist.append(ssim_val)

        if (it + 1) in snap_times:
            capture(f"iter {it + 1}", x0_hat=x0_hat, t_used=t_used)

        if (it + 1) % max(1, opt_steps // 8) == 0 or it == 0:
            print(
                f"  iter {it+1:4d}/{opt_steps}  "
                f"obs={loss_obs.item():.6f}  reg={loss_reg.item():.6f}  "
                f"total={loss.item():.6f}  t={t_used:4d}  "
                f"MAE={mae:.1f} m/s  SSIM={ssim_val:.4f}"
            )

    # =========================================================================
    # Figures
    # =========================================================================

    # --- Figure 1: optimization evolution ---
    # 3 rows: current velocity / prior estimate x̂₀ / error (μ - x̂₀)
    ncols  = 1 + len(hist_vel)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 8.5))
    axes = np.atleast_2d(axes).reshape(3, ncols)

    # column 0: reference
    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title("target v\n(model60)", fontsize=8); axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[0, 0].set_ylabel("μ (m/s)",    fontsize=9)
    axes[1, 0].set_ylabel("x̂₀ (m/s)",  fontsize=9)
    axes[2, 0].set_ylabel("μ − x̂₀",   fontsize=9)

    for j, (vel, x0_np, t_j, lbl) in enumerate(
        zip(hist_vel, hist_x0hat, hist_t, hist_labels)
    ):
        col   = j + 1
        t_str = f"t={t_j}" if t_j >= 0 else "—"

        # row 0: current velocity μ
        axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(f"{lbl}\n({t_str})", fontsize=7)
        axes[0, col].axis("off")

        # row 1: prior estimate x̂₀
        axes[1, col].imshow(x0_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[1, col].axis("off")

        # row 2: error map μ − x̂₀ (the guiding direction, reversed sign)
        diff    = vel - x0_np
        dlim    = max(float(np.abs(diff).max()), 1.0)
        axes[2, col].imshow(diff, cmap="coolwarm", aspect="auto", vmin=-dlim, vmax=dlim)
        axes[2, col].axis("off")

    plt.suptitle(
        f"RED-DiffEq evolution  |  init={init_type}  λ={reg_lambda}  "
        f"opt_steps={opt_steps}  model60[{sample_index}]\n"
        f"x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t  |  guiding force ∝ −(μ − x̂₀)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 2: final result ---
    final_vel_np = v_denormalize_np(mu)
    mae_final, ssim_final = velocity_mae_ssim(final_vel_np, target_vel_np, device, vel_vmin, vel_vmax)

    with torch.no_grad():
        wave_final = forward_wave(mu)

    err_v     = final_vel_np - target_vel_np
    err_lim_v = float(np.max(np.abs(err_v))) + 1e-6
    sh = wave_plot_shot
    tw = target_wave[sh].detach().cpu().numpy()
    pw = wave_final[sh].detach().cpu().numpy()
    ew = pw - tw
    wlim = max(np.abs(tw).max(), np.abs(pw).max()) + 1e-6
    elw  = float(np.abs(ew).max()) + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np, "target v (m/s)",  "viridis",  vel_vmin,   vel_vmax),
        (final_vel_np,  "pred v (m/s)",    "viridis",  vel_vmin,   vel_vmax),
        (err_v,         "v error (m/s)",   "coolwarm", -err_lim_v, err_lim_v),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
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
        f"RED-DiffEq  |  {init_type} init  λ={reg_lambda}  "
        f"MAE={mae_final:.1f} m/s  SSIM={ssim_final:.4f}  "
        f"obs={obs_losses[-1]:.6f}  reg={reg_losses[-1]:.6f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 3: metrics ---
    iters = list(range(1, opt_steps + 1))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))

    axes[0, 0].plot(iters, obs_losses,   color="C0"); axes[0, 0].set_title("Obs loss (wave)");    axes[0, 0].set_ylabel(wave_loss.upper()); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(iters, reg_losses,   color="C5"); axes[0, 1].set_title("RED reg loss (raw)"); axes[0, 1].set_ylabel("L_reg");           axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(iters, total_losses, color="C3"); axes[0, 2].set_title("Total loss");          axes[0, 2].set_ylabel("L_obs + λ·L_reg"); axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(iters, mae_hist,     color="C1"); axes[1, 0].set_title("Velocity MAE");        axes[1, 0].set_ylabel("MAE (m/s)");       axes[1, 0].set_xlabel("Iteration"); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(iters, ssim_hist,    color="C2"); axes[1, 1].set_title("Velocity SSIM");       axes[1, 1].set_ylabel("SSIM");            axes[1, 1].set_xlabel("Iteration"); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].scatter(iters, t_hist, s=1, alpha=0.3, color="C4")
    axes[1, 2].set_title("Sampled timestep t"); axes[1, 2].set_ylabel("t"); axes[1, 2].set_xlabel("Iteration"); axes[1, 2].grid(True, alpha=0.3)
    axes[2, 0].plot(iters, grad_obs_norms, color="C0"); axes[2, 0].set_title("‖∇_μ L_obs‖");       axes[2, 0].set_ylabel("Gradient norm"); axes[2, 0].set_xlabel("Iteration"); axes[2, 0].grid(True, alpha=0.3)
    axes[2, 1].plot(iters, grad_reg_norms, color="C5"); axes[2, 1].set_title("‖∇_μ L_reg‖ (raw)"); axes[2, 1].set_ylabel("Gradient norm"); axes[2, 1].set_xlabel("Iteration"); axes[2, 1].grid(True, alpha=0.3)
    axes[2, 2].plot(iters, grad_obs_norms, color="C0", label="‖∇L_obs‖")
    axes[2, 2].plot(iters, grad_reg_norms, color="C5", label="‖∇L_reg‖ (raw)", linestyle="--")
    axes[2, 2].set_title("Gradient norms (overlay)"); axes[2, 2].legend(fontsize=8)
    axes[2, 2].set_xlabel("Iteration"); axes[2, 2].grid(True, alpha=0.3)

    plt.suptitle(
        f"RED-DiffEq metrics  |  λ={reg_lambda}  init={init_type}  model60[{sample_index}]",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- summary ---
    summary = {
        "init_type": init_type,
        "final_obs_loss": float(obs_losses[-1]),
        "final_reg_loss": float(reg_losses[-1]),
        "final_total_loss": float(total_losses[-1]),
        "final_vel_mae_m_s": float(mae_final),
        "final_vel_ssim": float(ssim_final),
        "reg_lambda": reg_lambda,
        "max_timestep": max_timestep,
        "sigma_x0": sigma_x0,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {out_dir}")
    print(f"  MAE={mae_final:.1f} m/s  SSIM={ssim_final:.4f}")


if __name__ == "__main__":
    main()
