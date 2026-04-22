#!/usr/bin/env python3
"""
Batch evaluation of ddim_red_diffeq across CurveVel-B samples.

For each sample in [sample_start, sample_end), runs RED-DiffEq optimization and
records the final metric:
  1. mu  — optimized physical velocity field

Saves:
  summary.json           — per-sample metrics + aggregate mean/std/median
  metrics_per_sample.csv — flat CSV for downstream analysis
  metrics_aggregate.png  — box plots of MAE / SSIM across samples
  examples/              — full visualizations for example_indices

Run (from Manifold_constrained_FWI/):
  uv run python exp/DDIM/ddim_red_diffeq_batch/demo_ddim_red_diffeq_batch.py \\
    --config exp/DDIM/ddim_red_diffeq_batch/config_ddim_red_diffeq_batch.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
    from diffusers import DDPMScheduler
    from openfwi_unet_wrapper import load_openfwi_checkpoint
finally:
    sys.stderr.close()
    sys.stderr = _saved_py_stderr
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd2)

from src.core import pytorch_ssim
from src.seismic import seismic_master_forward_modeling


# ---------------------------------------------------------------------------
# helpers (same as ddim_red_diffeq)
# ---------------------------------------------------------------------------

def _resolve_path(p):
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_MANIFOLD_ROOT / path).resolve()

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def v_denormalize_np(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().numpy()
    return v.astype(np.float32).squeeze() * 1500.0 + 3000.0

def v_denormalize_tensor(v):
    return v * 1500.0 + 3000.0

def v_normalize_np(vel_np):
    return (vel_np / 1500.0 - 2.0).astype(np.float32)

def vel_to_ssim_tensor(vel_np, vmin, vmax):
    arr = np.squeeze(vel_np).astype(np.float32)
    t = torch.from_numpy(arr).view(1, 1, arr.shape[-2], arr.shape[-1])
    return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0)

def velocity_mae_ssim(pred_np, target_np, device, vmin, vmax):
    mae = float(np.mean(np.abs(pred_np - target_np)))
    t1  = vel_to_ssim_tensor(pred_np,   vmin, vmax).to(device)
    t2  = vel_to_ssim_tensor(target_np, vmin, vmax).to(device)
    ssim_val = float(pytorch_ssim.ssim(t1, t2, window_size=11, size_average=True).item())
    return mae, ssim_val

def load_model(checkpoint, training_yaml):
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

def forward_wave(mu):
    v_phys = v_denormalize_tensor(mu.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)

def red_reg_loss(wrapper, alphas_cumprod, mu, max_timestep, sigma_x0,
                 use_time_weight, rng):
    if sigma_x0 > 0:
        x0_pred = mu + sigma_x0 * torch.randn(mu.shape, device=mu.device, generator=rng)
    else:
        x0_pred = mu
    t_val    = int(torch.randint(0, max_timestep, (1,), generator=rng, device=mu.device).item())
    t_tensor = torch.tensor([t_val], device=mu.device, dtype=torch.long)
    alpha_bar = alphas_cumprod[t_val].to(mu.device)
    eps       = torch.randn(x0_pred.shape, device=mu.device, generator=rng)
    x_t       = alpha_bar.sqrt() * x0_pred + (1 - alpha_bar).sqrt() * eps
    with torch.no_grad():
        eps_pred = wrapper(x_t.detach(), t_tensor).sample
    gradient_field = (eps_pred - eps).detach()
    if use_time_weight:
        w_t = ((1 - alpha_bar) / alpha_bar.clamp(min=1e-8)).sqrt()
        gradient_field = gradient_field * w_t
    loss = (gradient_field * x0_pred).mean()
    return loss, t_val

def build_init(init_type, target_vel_np, smooth_sigma, device):
    v_np = v_normalize_np(target_vel_np)
    if init_type == "smoothed":
        mu_np = gaussian_filter(v_np, sigma=smooth_sigma).astype(np.float32)
    elif init_type == "homogeneous":
        mu_np = np.full(v_np.shape, float(np.min(v_np[0, :])), dtype=np.float32)
    elif init_type == "linear":
        v_min, v_max = float(np.min(v_np)), float(np.max(v_np))
        depth_grad = np.linspace(v_min, v_max, v_np.shape[0], dtype=np.float32)
        mu_np = np.tile(depth_grad.reshape(-1, 1), (1, v_np.shape[1]))
    else:  # "zero"
        mu_np = np.zeros(v_np.shape, dtype=np.float32)
    mu = torch.from_numpy(mu_np).reshape(1, 1, 70, 70).to(device)
    return mu.clamp(-1.0, 1.0).requires_grad_(True)

def save_example_figure(sample_idx, target_vel_np, mu_np,
                        mae, ssim_val, vel_vmin, vel_vmax, out_path):
    err   = mu_np - target_vel_np
    elim  = max(float(np.abs(err).max()), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes, [
        (target_vel_np, "target v",                                   "viridis",  vel_vmin,  vel_vmax),
        (mu_np,         f"μ (RED-DiffEq)\nMAE={mae:.0f}  SSIM={ssim_val:.3f}", "viridis",  vel_vmin,  vel_vmax),
        (err,           "error (m/s)",                                "coolwarm", -elim,     elim),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=8); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Sample {sample_idx} | CurveVel-B | RED-DiffEq", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(_SCRIPT_DIR / "config_ddim_red_diffeq_batch.yaml"))
    args = parser.parse_args()
    cfg  = load_config(Path(args.config).resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device = torch.device(cfg.get("device", "cuda:0"))
    cp.cuda.Device(device.index if device.index is not None else 0).use()

    base_seed = int(cfg.get("seed", 42))

    # --- model ---
    dcfg          = cfg.get("ddim") or {}
    ckpt          = _resolve_path(dcfg["checkpoint"])
    training_yaml = _TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml"

    wrapper, ddpm_sch = load_model(ckpt, training_yaml)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)
    alphas_cumprod = ddpm_sch.alphas_cumprod.to(device)

    # --- optimization params (read once) ---
    opt_cfg       = cfg.get("optimization") or {}
    opt_steps     = int(opt_cfg.get("opt_steps", 300))
    lr            = float(opt_cfg.get("lr", 0.03))
    reg_lambda    = float(opt_cfg.get("reg_lambda", 0.75))
    wave_loss_type = str(opt_cfg.get("wave_loss", "l1")).lower()

    red_cfg         = cfg.get("red") or {}
    max_timestep    = int(red_cfg.get("max_timestep", 1000))
    sigma_x0        = float(red_cfg.get("sigma_x0", 0.0001))
    use_time_weight = bool(red_cfg.get("use_time_weight", False))

    init_cfg     = cfg.get("initialization") or {}
    init_type    = str(init_cfg.get("type", "smoothed"))
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))

    viz      = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))

    batch_cfg       = cfg.get("batch") or {}
    sample_start    = int(batch_cfg.get("sample_start", 0))
    sample_end      = int(batch_cfg.get("sample_end", 100))
    example_indices = set(int(x) for x in batch_cfg.get("example_indices", [0, 5, 10, 25, 50]))

    cv_cfg     = cfg.get("curvevel") or {}
    model_path = _resolve_path(cv_cfg["model60_path"])
    if not model_path.is_file():
        raise SystemExit(f"Data not found: {model_path}")
    all_data = np.load(model_path)  # (N, 1, 70, 70)

    # --- output dir ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_red_diffeq_batch_{ts}"
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)

    n_samples = sample_end - sample_start
    print(
        f"[Batch RED-DiffEq | CurveVel-B]  samples={sample_start}–{sample_end - 1}  "
        f"opt_steps={opt_steps}  lr={lr}  λ={reg_lambda}  init={init_type}  device={device}",
        flush=True,
    )

    _print_every = max(1, opt_steps // 4)

    # --- CSV log ---
    csv_path = out_dir / "metrics_per_sample.csv"
    csv_file  = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["sample_idx", "mu_mae", "mu_ssim"])

    records: list[dict] = []
    t_batch_start = time.time()

    for si, sample_idx in enumerate(range(sample_start, sample_end)):
        t0 = time.time()
        print(f"\n[{si+1:3d}/{n_samples}] sample={sample_idx}  optimization starting ...", flush=True)

        target_vel_np = all_data[sample_idx, 0].astype(np.float32)

        with torch.no_grad():
            target_wave = seismic_master_forward_modeling(
                torch.from_numpy(target_vel_np).float().to(device)
            )

        # ── initialize μ and RNG ──────────────────────────────────────────────
        mu  = build_init(init_type, target_vel_np, smooth_sigma, device)
        rng = torch.Generator(device=device).manual_seed(base_seed + sample_idx)

        # ── optimizer ────────────────────────────────────────────────────────
        optimizer = torch.optim.Adam([mu], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt_steps, eta_min=0.0
        )

        # ── optimization loop ─────────────────────────────────────────────────
        for it in range(opt_steps):
            optimizer.zero_grad(set_to_none=True)
            wave_pred = forward_wave(mu)
            loss_obs  = (F.l1_loss(wave_pred, target_wave) if wave_loss_type == "l1"
                         else F.mse_loss(wave_pred, target_wave))
            loss_reg, t_used = red_reg_loss(
                wrapper, alphas_cumprod, mu,
                max_timestep, sigma_x0, use_time_weight, rng,
            )
            loss = loss_obs + reg_lambda * loss_reg
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                mu.data.clamp_(-1.0, 1.0)

            if (it + 1) % _print_every == 0:
                print(f"  iter {it+1:4d}/{opt_steps}  "
                      f"obs={loss_obs.item():.5f}  reg={loss_reg.item():.5f}  "
                      f"t={time.time()-t0:.0f}s", flush=True)

        # ── final metrics ─────────────────────────────────────────────────────
        with torch.no_grad():
            mu_np = v_denormalize_np(mu)
        mae, ssim_val = velocity_mae_ssim(mu_np, target_vel_np, device, vel_vmin, vel_vmax)

        elapsed       = time.time() - t0
        elapsed_total = time.time() - t_batch_start
        eta_s         = elapsed_total / (si + 1) * (n_samples - si - 1)
        print(
            f"  => sample={sample_idx:3d} DONE  "
            f"μ: MAE={mae:.0f} SSIM={ssim_val:.3f}  "
            f"t={elapsed:.0f}s  ETA={eta_s/60:.1f}min",
            flush=True,
        )

        records.append({"sample_idx": sample_idx, "mu_mae": mae, "mu_ssim": ssim_val})
        csv_writer.writerow([sample_idx, f"{mae:.4f}", f"{ssim_val:.6f}"])
        csv_file.flush()

        if sample_idx in example_indices:
            save_example_figure(
                sample_idx, target_vel_np, mu_np,
                mae, ssim_val, vel_vmin, vel_vmax,
                out_dir / "examples" / f"sample_{sample_idx:03d}.png",
            )

    csv_file.close()

    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    agg = {}
    for k in ["mu_mae", "mu_ssim"]:
        vals = [r[k] for r in records]
        agg[k] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "median": float(np.median(vals)),
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
        }

    summary = {
        "config": {
            "sample_start": sample_start, "sample_end": sample_end,
            "opt_steps": opt_steps, "init_type": init_type,
        },
        "aggregate": agg,
        "per_sample": records,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── print aggregate table ─────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print(f"{'Method':<14}  {'MAE mean±std':>18}  {'SSIM mean±std':>18}  {'median MAE':>12}")
    print("-"*60)
    m, s = agg["mu_mae"], agg["mu_ssim"]
    print(f"{'μ (RED-DiffEq)':<14}  "
          f"{m['mean']:8.1f} ± {m['std']:6.1f} m/s  "
          f"{s['mean']:6.4f} ± {s['std']:6.4f}  "
          f"{m['median']:10.1f} m/s")
    print("="*60)

    # =========================================================================
    # Aggregate figure: box plots
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    mae_data  = [[r["mu_mae"]  for r in records]]
    ssim_data = [[r["mu_ssim"] for r in records]]
    labels    = ["μ (RED-DiffEq)"]

    bp1 = axes[0].boxplot(mae_data, labels=labels, patch_artist=True)
    bp1["boxes"][0].set_facecolor("#4C72B0"); bp1["boxes"][0].set_alpha(0.7)
    axes[0].set_ylabel("MAE (m/s)"); axes[0].set_title("Velocity MAE")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].text(1, np.mean(mae_data[0]), f"{np.mean(mae_data[0]):.0f}",
                 ha="center", va="bottom", fontsize=9)

    bp2 = axes[1].boxplot(ssim_data, labels=labels, patch_artist=True)
    bp2["boxes"][0].set_facecolor("#4C72B0"); bp2["boxes"][0].set_alpha(0.7)
    axes[1].set_ylabel("SSIM"); axes[1].set_title("Velocity SSIM")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].text(1, np.mean(ssim_data[0]), f"{np.mean(ssim_data[0]):.4f}",
                 ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        f"RED-DiffEq  |  CurveVel-B  samples {sample_start}–{sample_end-1}  "
        f"(n={n_samples})  steps={opt_steps}  init={init_type}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_aggregate.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutput: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
