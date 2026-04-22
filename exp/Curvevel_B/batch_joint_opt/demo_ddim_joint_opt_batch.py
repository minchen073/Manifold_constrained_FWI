#!/usr/bin/env python3
"""
Batch evaluation of ddim_joint_opt across CurveVel-B samples.

For each sample in [sample_start, sample_end), runs Phase 1 + Phase 2 and
records three final metrics:
  1. v      — Phase 1 physical velocity (FWI result)
  2. v_gen  — Phase 1 z→DDIM generation
  3. v_p2   — Phase 2 z→DDIM generation

Saves:
  summary.json           — per-sample metrics + aggregate mean/std/median
  metrics_aggregate.png  — box plots of MAE / SSIM across samples
  examples/              — full visualizations for example_indices

Run (from Manifold_constrained_FWI/):
  uv run python exp/Curvevel_B/ddim_joint_opt_batch/demo_ddim_joint_opt_batch.py \\
    --config exp/Curvevel_B/ddim_joint_opt_batch/config_ddim_joint_opt_batch.yaml
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
# helpers (same as ddim_joint_opt)
# ---------------------------------------------------------------------------

def _resolve_path(p):
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_MANIFOLD_ROOT / path).resolve()

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def v_normalize_np(vel_np):
    return (vel_np / 1500.0 - 2.0).astype(np.float32)

def v_denormalize_np(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().numpy()
    return v.astype(np.float32).squeeze() * 1500.0 + 3000.0

def v_denormalize_tensor(v):
    return v * 1500.0 + 3000.0

def vel_to_ssim_tensor(vel_np, vmin, vmax):
    arr = np.squeeze(vel_np).astype(np.float32)
    t = torch.from_numpy(arr).view(1, 1, arr.shape[-2], arr.shape[-1])
    return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0)

def velocity_mae_ssim(pred_np, target_np, device, vmin, vmax):
    mae = float(np.mean(np.abs(pred_np - target_np)))
    t1  = vel_to_ssim_tensor(pred_np,  vmin, vmax).to(device)
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
            tcfg = yaml.safe_load(f)
        s = str((tcfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        torch_dtype = torch.float16 if s in ("float16", "fp16") else torch.float32
    wrapper  = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)
    ddpm_sch = DDPMScheduler.from_pretrained(str(checkpoint))
    ddim_sch = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
    return wrapper, ddpm_sch, ddim_sch

def sample_with_grad(wrapper, ddim_sched, z, num_steps, eta=0.0):
    ddim_sched.set_timesteps(num_steps)
    x = z
    for t in ddim_sched.timesteps:
        model_output = wrapper(x, t).sample
        x = ddim_sched.step(model_output, t, x, eta=eta,
                            use_clipped_model_output=False).prev_sample
    return x

def sample_no_grad(wrapper, ddim_sched, z, num_steps, eta=0.0):
    with torch.no_grad():
        return sample_with_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)

def build_ddim_timesteps(t_start, num_steps):
    return [round(t_start * (num_steps - 1 - i) / max(1, num_steps - 1))
            for i in range(num_steps)]

def _ddim_step(model_output, t, t_prev, sample, alphas_cumprod,
               final_alpha_cumprod=1.0, eta=0.0):
    device   = sample.device
    acp_t    = alphas_cumprod[t].to(device)
    acp_prev = (alphas_cumprod[t_prev].to(device) if t_prev >= 0
                else torch.tensor(final_alpha_cumprod, dtype=torch.float32, device=device))
    beta_t   = 1.0 - acp_t
    pred_x0  = (sample - beta_t.sqrt() * model_output) / acp_t.sqrt()
    variance = ((1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - acp_t / acp_prev)).clamp(min=0.0)
    sigma    = eta * variance.sqrt()
    coef_dir = (1.0 - acp_prev - sigma ** 2).clamp(min=0.0).sqrt()
    prev_sample = acp_prev.sqrt() * pred_x0 + coef_dir * model_output
    if eta > 0.0:
        prev_sample = prev_sample + sigma * torch.randn_like(sample)
    return prev_sample

def sample_custom_with_grad(wrapper, alphas_cumprod, z, t_start, num_steps,
                             eta=0.0, final_alpha_cumprod=1.0):
    timesteps = build_ddim_timesteps(t_start, num_steps)
    x = z
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        model_output = wrapper(x, torch.tensor(t, device=z.device)).sample
        x = _ddim_step(model_output, t, t_prev, x, alphas_cumprod, final_alpha_cumprod, eta)
    return x

def sample_custom_no_grad(wrapper, alphas_cumprod, z, t_start, num_steps,
                           eta=0.0, final_alpha_cumprod=1.0):
    with torch.no_grad():
        return sample_custom_with_grad(wrapper, alphas_cumprod, z.detach(),
                                       t_start, num_steps, eta, final_alpha_cumprod)

def forward_wave(v):
    v_phys = v_denormalize_tensor(v.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)

def build_v_init(target_vel_np, smooth_sigma, device):
    v_np  = v_normalize_np(target_vel_np)
    mu_np = gaussian_filter(v_np, sigma=smooth_sigma).astype(np.float32)
    mu    = torch.from_numpy(mu_np).reshape(1, 1, 70, 70).to(device)
    return mu.clamp(-1.0, 1.0).requires_grad_(True)

def compute_lambda(step, warmup_steps, ramp_steps, lambda_max):
    if step < warmup_steps:
        return 0.0
    t = min(step - warmup_steps, ramp_steps) / max(1, ramp_steps)
    return float(lambda_max * t)

def make_scheduler(opt, sched_type, t_max, eta_min):
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    return None

def make_optimizer(params, opt_type, lr):
    if opt_type == "gd":
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


# ---------------------------------------------------------------------------
# example figure
# ---------------------------------------------------------------------------

def save_example_figure(
    sample_idx, target_vel_np,
    v_p1_np, v_gen_np, v_p2_np,
    mae_v, ssim_v, mae_gen, ssim_gen, mae_p2, ssim_p2,
    vel_vmin, vel_vmax, out_path,
):
    err_v   = v_p1_np  - target_vel_np
    err_gen = v_gen_np - target_vel_np
    err_p2  = v_p2_np  - target_vel_np
    elim    = max(float(np.abs(err_v).max()), float(np.abs(err_gen).max()),
                  float(np.abs(err_p2).max()), 1e-6)

    cols = [
        (target_vel_np, "target",
         f"v (P1)\nMAE={mae_v:.0f} m/s  SSIM={ssim_v:.3f}",
         f"v_gen (P1)\nMAE={mae_gen:.0f} m/s  SSIM={ssim_gen:.3f}",
         f"v_p2 (P2)\nMAE={mae_p2:.0f} m/s  SSIM={ssim_p2:.3f}"),
    ]
    vels  = [target_vel_np, v_p1_np, v_gen_np, v_p2_np]
    errs  = [None,          err_v,   err_gen,  err_p2]
    titles_top = ["target v", f"v (P1)\nMAE={mae_v:.0f}  SSIM={ssim_v:.3f}",
                  f"v_gen (P1)\nMAE={mae_gen:.0f}  SSIM={ssim_gen:.3f}",
                  f"v_p2 (P2)\nMAE={mae_p2:.0f}  SSIM={ssim_p2:.3f}"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for j, (vel, title) in enumerate(zip(vels, titles_top)):
        im = axes[0, j].imshow(vel, cmap="viridis", aspect="auto",
                               vmin=vel_vmin, vmax=vel_vmax)
        axes[0, j].set_title(title, fontsize=8); axes[0, j].axis("off")
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)

    axes[1, 0].axis("off")
    for j, err in enumerate(errs[1:], start=1):
        im = axes[1, j].imshow(err, cmap="coolwarm", aspect="auto",
                               vmin=-elim, vmax=elim)
        axes[1, j].set_title("error (m/s)", fontsize=8); axes[1, j].axis("off")
        plt.colorbar(im, ax=axes[1, j], fraction=0.046)

    plt.suptitle(f"Sample {sample_idx} | CurveVel-B", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(_SCRIPT_DIR / "config_ddim_joint_opt_batch.yaml"))
    args   = parser.parse_args()
    cfg    = load_config(Path(args.config).resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str)
    cp.cuda.Device(device.index if device.index is not None else 0).use()

    base_seed = int(cfg.get("seed", 42))

    # --- model ---
    dcfg  = cfg.get("ddim") or {}
    ckpt  = _resolve_path(dcfg["checkpoint"])
    eta   = float(dcfg.get("eta", 0.0))
    training_yaml = _TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml"

    wrapper, ddpm_sch, ddim_sch = load_model(ckpt, training_yaml)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)

    # --- optimization params (read once) ---
    opt_cfg          = cfg.get("optimization") or {}
    total_steps      = int(opt_cfg.get("total_steps", 300))
    lr_v             = float(opt_cfg.get("lr_v", 0.03))
    lr_z             = float(opt_cfg.get("lr_z", 0.02))
    z_steps_per_iter = int(opt_cfg.get("z_steps_per_iter", 1))
    wave_loss_type   = str(opt_cfg.get("wave_loss", "l1")).lower()
    num_steps        = int(opt_cfg.get("num_steps", 6))
    schedv_cfg     = opt_cfg.get("scheduler_v") or {}
    schedv_type    = str(schedv_cfg.get("type", "cosine")).lower()
    schedv_eta_min = float(schedv_cfg.get("eta_min", 0.0))
    schedz_cfg     = opt_cfg.get("scheduler_z") or {}
    schedz_type    = str(schedz_cfg.get("type", "cosine")).lower()
    schedz_eta_min = float(schedz_cfg.get("eta_min", 0.0))
    optim_v_type   = str(opt_cfg.get("optimizer_v", "adam")).lower()
    optim_z_type   = str(opt_cfg.get("optimizer_z", "adam")).lower()

    guide_cfg    = cfg.get("guidance") or {}
    warmup_steps = int(guide_cfg.get("warmup_steps", 50))
    ramp_steps   = int(guide_cfg.get("ramp_steps", 100))
    lambda_max   = float(guide_cfg.get("lambda_max", 0.75))

    p2_cfg        = cfg.get("phase2") or {}
    p2_steps      = int(p2_cfg.get("opt_steps", 100))
    p2_lr         = float(p2_cfg.get("lr", 0.002))
    p2_nsteps     = int(p2_cfg.get("num_steps", num_steps))
    p2_optim_type = str(p2_cfg.get("optimizer", "adam")).lower()
    p2_sched_cfg  = p2_cfg.get("scheduler") or {}
    p2_sched_type = str(p2_sched_cfg.get("type", "cosine")).lower()
    p2_eta_min    = float(p2_sched_cfg.get("eta_min", 0.0))

    init_cfg     = cfg.get("initialization") or {}
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
    out_dir = out_base.resolve() / f"ddim_joint_opt_batch_{ts}"
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)

    log_path = out_dir / "metrics_per_sample.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["sample_idx",
                          "v_mae", "v_ssim",
                          "vgen_mae", "vgen_ssim",
                          "p2_mae", "p2_ssim"])
    log_file.flush()

    # --- DDIM timestep metadata (computed once) ---
    ddim_sch.set_timesteps(num_steps)
    t_start             = int(ddim_sch.timesteps[0].item())
    alphas_cumprod      = ddpm_sch.alphas_cumprod
    final_alpha_cumprod = float(ddim_sch.final_alpha_cumprod)

    n_samples = sample_end - sample_start
    print(
        f"[Batch | CurveVel-B]  samples={sample_start}–{sample_end - 1}  "
        f"P1_steps={total_steps}  P2_steps={p2_steps}  "
        f"num_steps(P1/P2)={num_steps}/{p2_nsteps}  device={device}",
        flush=True,
    )
    _p1_print_every = max(1, total_steps  // 4)
    _p2_print_every = max(1, p2_steps // 2)

    # --- accumulators ---
    records: list[dict] = []

    t_batch_start = time.time()

    for si, sample_idx in enumerate(range(sample_start, sample_end)):
        t0 = time.time()
        print(f"\n[{si+1:3d}/{n_samples}] sample={sample_idx}  Phase 1 starting ...", flush=True)
        target_vel_np = all_data[sample_idx, 0].astype(np.float32)

        with torch.no_grad():
            target_wave = seismic_master_forward_modeling(
                torch.from_numpy(target_vel_np).float().to(device)
            )

        # ── initialize v and z ────────────────────────────────────────────────
        v = build_v_init(target_vel_np, smooth_sigma, device)
        z = torch.randn(1, 1, 70, 70, device=device, dtype=torch.float32,
                        generator=torch.Generator(device=device).manual_seed(base_seed + sample_idx))
        z = z.requires_grad_(True)

        # ── Phase 1 optimizers ────────────────────────────────────────────────
        opt_v   = make_optimizer([v], optim_v_type, lr_v)
        sched_v = make_scheduler(opt_v, schedv_type, total_steps, schedv_eta_min)
        opt_z   = make_optimizer([z], optim_z_type, lr_z)
        sched_z = make_scheduler(opt_z, schedz_type, total_steps, schedz_eta_min)

        # ── Phase 1 loop ──────────────────────────────────────────────────────
        for it in range(total_steps):
            for _ in range(z_steps_per_iter):
                opt_z.zero_grad(set_to_none=True)
                v_gen_z = sample_with_grad(wrapper, ddim_sch, z, num_steps, eta)
                F.mse_loss(v_gen_z, v.detach()).backward()
                opt_z.step()
            if sched_z is not None: sched_z.step()

            v_gen = sample_no_grad(wrapper, ddim_sch, z, num_steps, eta)

            lam = compute_lambda(it, warmup_steps, ramp_steps, lambda_max)
            opt_v.zero_grad(set_to_none=True)
            wave_pred = forward_wave(v)
            loss_wave = (F.l1_loss(wave_pred, target_wave) if wave_loss_type == "l1"
                         else F.mse_loss(wave_pred, target_wave))
            loss_v = (loss_wave + lam * F.mse_loss(v, v_gen.detach())
                      if lam > 0.0 else loss_wave)
            loss_v.backward()
            opt_v.step()
            if sched_v is not None: sched_v.step()
            v.data.clamp_(-1.0, 1.0)

            if (it + 1) % _p1_print_every == 0:
                print(f"  P1 iter {it+1:4d}/{total_steps}  "
                      f"wave={loss_wave.item():.5f}  λ={lam:.3f}  "
                      f"t={time.time()-t0:.0f}s", flush=True)

        # ── Phase 1 final metrics ─────────────────────────────────────────────
        with torch.no_grad():
            v_p1_np  = v_denormalize_np(v)
            v_gen_np = v_denormalize_np(sample_no_grad(wrapper, ddim_sch, z, num_steps, eta))
        mae_v,   ssim_v   = velocity_mae_ssim(v_p1_np,  target_vel_np, device, vel_vmin, vel_vmax)
        mae_gen, ssim_gen = velocity_mae_ssim(v_gen_np, target_vel_np, device, vel_vmin, vel_vmax)

        # ── Phase 2 optimizers ────────────────────────────────────────────────
        print(f"  Phase 2 starting ...", flush=True)
        opt_z2   = make_optimizer([z], p2_optim_type, p2_lr)
        sched_z2 = make_scheduler(opt_z2, p2_sched_type, max(1, p2_steps), p2_eta_min)

        # ── Phase 2 loop ──────────────────────────────────────────────────────
        for it2 in range(p2_steps):
            opt_z2.zero_grad(set_to_none=True)
            pred_n  = sample_custom_with_grad(
                wrapper, alphas_cumprod, z, t_start, p2_nsteps, eta, final_alpha_cumprod
            )
            loss_p2 = F.mse_loss(forward_wave(pred_n), target_wave)
            loss_p2.backward()
            opt_z2.step()
            if sched_z2 is not None: sched_z2.step()

            if (it2 + 1) % _p2_print_every == 0:
                print(f"  P2 iter {it2+1:4d}/{p2_steps}  "
                      f"wave={loss_p2.item():.5f}  "
                      f"t={time.time()-t0:.0f}s", flush=True)

        # ── Phase 2 final metrics ─────────────────────────────────────────────
        with torch.no_grad():
            v_p2_np = v_denormalize_np(
                sample_custom_no_grad(wrapper, alphas_cumprod, z, t_start,
                                      p2_nsteps, eta, final_alpha_cumprod)
            )
        mae_p2, ssim_p2 = velocity_mae_ssim(v_p2_np, target_vel_np, device, vel_vmin, vel_vmax)

        elapsed = time.time() - t0
        elapsed_total = time.time() - t_batch_start
        eta_s = elapsed_total / (si + 1) * (n_samples - si - 1)
        print(
            f"  => sample={sample_idx:3d} DONE  "
            f"v: MAE={mae_v:.0f} SSIM={ssim_v:.3f}  "
            f"v_gen: MAE={mae_gen:.0f} SSIM={ssim_gen:.3f}  "
            f"v_p2: MAE={mae_p2:.0f} SSIM={ssim_p2:.3f}  "
            f"t={elapsed:.0f}s  ETA={eta_s/60:.1f}min",
            flush=True,
        )

        records.append({
            "sample_idx": sample_idx,
            "v_mae":    mae_v,   "v_ssim":    ssim_v,
            "vgen_mae": mae_gen, "vgen_ssim": ssim_gen,
            "p2_mae":   mae_p2,  "p2_ssim":   ssim_p2,
        })
        log_writer.writerow([sample_idx,
                              f"{mae_v:.4f}",   f"{ssim_v:.6f}",
                              f"{mae_gen:.4f}", f"{ssim_gen:.6f}",
                              f"{mae_p2:.4f}",  f"{ssim_p2:.6f}"])
        log_file.flush()

        # ── example figure ─────────────────────────────────────────────────────
        if sample_idx in example_indices:
            save_example_figure(
                sample_idx, target_vel_np, v_p1_np, v_gen_np, v_p2_np,
                mae_v, ssim_v, mae_gen, ssim_gen, mae_p2, ssim_p2,
                vel_vmin, vel_vmax,
                out_dir / "examples" / f"sample_{sample_idx:03d}.png",
            )

    log_file.close()

    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    keys = ["v_mae", "v_ssim", "vgen_mae", "vgen_ssim", "p2_mae", "p2_ssim"]
    agg  = {}
    for k in keys:
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
            "total_steps": total_steps, "p2_steps": p2_steps,
            "num_steps_p1": num_steps, "num_steps_p2": p2_nsteps,
        },
        "aggregate": agg,
        "per_sample": records,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── print aggregate table ─────────────────────────────────────────────────
    print("\n" + "="*70, flush=True)
    print(f"{'Method':<12}  {'MAE mean±std':>18}  {'SSIM mean±std':>18}  {'median MAE':>12}")
    print("-"*70)
    for method, mae_k, ssim_k in [
        ("v (P1)",     "v_mae",    "v_ssim"),
        ("v_gen (P1)", "vgen_mae", "vgen_ssim"),
        ("v_p2 (P2)",  "p2_mae",   "p2_ssim"),
    ]:
        m, s = agg[mae_k], agg[ssim_k]
        print(f"{method:<12}  "
              f"{m['mean']:8.1f} ± {m['std']:6.1f} m/s  "
              f"{s['mean']:6.4f} ± {s['std']:6.4f}  "
              f"{m['median']:10.1f} m/s")
    print("="*70)

    # =========================================================================
    # Aggregate figure: box plots
    # =========================================================================
    mae_data  = [[r["v_mae"]    for r in records],
                 [r["vgen_mae"] for r in records],
                 [r["p2_mae"]   for r in records]]
    ssim_data = [[r["v_ssim"]    for r in records],
                 [r["vgen_ssim"] for r in records],
                 [r["p2_ssim"]   for r in records]]
    labels = ["v (P1)", "v_gen (P1)", "v_p2 (P2)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    bp1 = axes[0].boxplot(mae_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], ["#4C72B0", "#DD8452", "#55A868"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_ylabel("MAE (m/s)"); axes[0].set_title("Velocity MAE")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(mae_data, 1):
        axes[0].text(i, np.mean(d), f"{np.mean(d):.0f}", ha="center",
                     va="bottom", fontsize=8, color="black")

    bp2 = axes[1].boxplot(ssim_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp2["boxes"], ["#4C72B0", "#DD8452", "#55A868"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].set_ylabel("SSIM"); axes[1].set_title("Velocity SSIM")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(ssim_data, 1):
        axes[1].text(i, np.mean(d), f"{np.mean(d):.4f}", ha="center",
                     va="bottom", fontsize=8, color="black")

    plt.suptitle(
        f"CurveVel-B  samples {sample_start}–{sample_end-1}  "
        f"(n={n_samples})  P1={total_steps} iters  P2={p2_steps} iters",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_aggregate.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
