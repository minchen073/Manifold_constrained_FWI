#!/usr/bin/env python3
"""
Joint Alternating Optimization: Physical FWI (v) + Latent Prior (z).
FlatFault-B 版本

Two-variable alternating scheme:

  v  (auxiliary, physical space)
     - Init: Gaussian-smoothed velocity model
     - Loss: L_wave(v) + λ(i) · L_guide(v; v_gen)
     - Guidance: direct L2 pull toward v_gen:
           L_guide = ||v − v_gen.detach()||²
     - λ warmup: 0 for first warmup_steps, then linear ramp to λ_max

  z  (latent code, manifold prior)
     - Init: z ~ N(0, I)
     - Loss: MSE(DDIM(z), v.detach())
     - Learns to generate velocity fields consistent with v's evolving shape

Per outer iteration:
  1. Update z for z_steps_per_iter steps  (target = v.detach())
  2. Decode v_gen = DDIM(z) with no_grad  (for v guidance)
  3. Update v for 1 step                  (wave loss + λ·guidance from v_gen)

与 Curvevel_B 版本的差异：
  - load_model 支持独立 model.pt 文件（无需 scheduler/ 子目录）
  - config 中用 flatfault.data_path 替代 curvevel.model60_path
  - 标题/日志标注 FlatFault-B

Run (from Manifold_constrained_FWI/):
  uv run python exp/FlatFault_B/ddim_joint_opt/demo_ddim_joint_opt.py \\
    --config exp/FlatFault_B/ddim_joint_opt/config_ddim_joint_opt.yaml
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
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR  = _MANIFOLD_ROOT / "training"
_CURVE_VEL_B   = _MANIFOLD_ROOT.parent / "CurveVelB"  # diffusers_torch_compat 所在目录

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


def v_normalize_np(vel_np: np.ndarray) -> np.ndarray:
    """Physical velocity m/s → [-1, 1]."""
    return (vel_np / 1500.0 - 2.0).astype(np.float32)


def v_denormalize_np(v: torch.Tensor | np.ndarray) -> np.ndarray:
    """[-1, 1] → physical velocity m/s, returns numpy."""
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().numpy()
    return v.astype(np.float32).squeeze() * 1500.0 + 3000.0


def v_denormalize_tensor(v: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → physical velocity m/s, keeps gradient."""
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
    t1  = vel_to_ssim_tensor(pred_np,  vmin, vmax).to(device)
    t2  = vel_to_ssim_tensor(target_np, vmin, vmax).to(device)
    ssim_val = float(pytorch_ssim.ssim(t1, t2, window_size=11, size_average=True).item())
    return mae, ssim_val


def load_model(model_path: Path, sched_cfg: dict | None = None):
    """Load OpenFWIUNetWrapper + DDPMScheduler + DDIMScheduler.

    model_path 可以是：
      - 包含 model.pt + scheduler/ 的 checkpoint 目录
      - 直接指向独立 model.pt 文件（此时 scheduler 由 sched_cfg 构建）
    """
    model_path = model_path.resolve()
    if model_path.is_dir():
        model_pt = model_path / "model.pt"
        if not model_pt.is_file():
            raise SystemExit(f"Missing {model_pt}")
        wrapper  = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch.float32)
        ddpm_sch = DDPMScheduler.from_pretrained(str(model_path), subfolder="scheduler")
        ddim_sch = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
    elif model_path.is_file() and model_path.suffix == ".pt":
        if sched_cfg is None:
            sched_cfg = {}
        ddpm_sch = DDPMScheduler(
            num_train_timesteps=int(sched_cfg.get("num_train_timesteps", 1000)),
            beta_start=float(sched_cfg.get("beta_start", 0.0001)),
            beta_end=float(sched_cfg.get("beta_end", 0.02)),
            beta_schedule=sched_cfg.get("beta_schedule", "linear"),
            prediction_type=sched_cfg.get("prediction_type", "epsilon"),
            timestep_spacing=sched_cfg.get("timestep_spacing", "leading"),
            clip_sample=False,
        )
        ddim_sch = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
        wrapper  = load_openfwi_checkpoint(model_path, map_location="cpu", torch_dtype=torch.float32)
    else:
        raise SystemExit(f"model_path 不存在或格式不支持: {model_path}")
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
    return x  # (1, 1, 70, 70)


def sample_no_grad(
    wrapper, ddim_sched, z: torch.Tensor, num_steps: int, eta: float = 0.0
) -> torch.Tensor:
    with torch.no_grad():
        return sample_with_grad(wrapper, ddim_sched, z.detach(), num_steps, eta)


# ---------------------------------------------------------------------------
# Custom DDIM sampler — fixed t_start, variable num_steps
# ---------------------------------------------------------------------------

def build_ddim_timesteps(t_start: int, num_steps: int) -> list[int]:
    return [
        round(t_start * (num_steps - 1 - i) / max(1, num_steps - 1))
        for i in range(num_steps)
    ]


def _ddim_step(
    model_output: torch.Tensor,
    t: int,
    t_prev: int,
    sample: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    final_alpha_cumprod: float = 1.0,
    eta: float = 0.0,
) -> torch.Tensor:
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


def sample_custom_with_grad(
    wrapper,
    alphas_cumprod: torch.Tensor,
    z: torch.Tensor,
    t_start: int,
    num_steps: int,
    eta: float = 0.0,
    final_alpha_cumprod: float = 1.0,
) -> torch.Tensor:
    timesteps = build_ddim_timesteps(t_start, num_steps)
    x = z
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        model_output = wrapper(x, torch.tensor(t, device=z.device)).sample
        x = _ddim_step(model_output, t, t_prev, x, alphas_cumprod, final_alpha_cumprod, eta)
    return x


def sample_custom_no_grad(
    wrapper,
    alphas_cumprod: torch.Tensor,
    z: torch.Tensor,
    t_start: int,
    num_steps: int,
    eta: float = 0.0,
    final_alpha_cumprod: float = 1.0,
) -> torch.Tensor:
    with torch.no_grad():
        return sample_custom_with_grad(
            wrapper, alphas_cumprod, z.detach(),
            t_start, num_steps, eta, final_alpha_cumprod,
        )


# ---------------------------------------------------------------------------
# wave forward operator
# ---------------------------------------------------------------------------

def forward_wave(v: torch.Tensor) -> torch.Tensor:
    v_phys = v_denormalize_tensor(v.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


# ---------------------------------------------------------------------------
# guidance
# ---------------------------------------------------------------------------

def guidance_from_vgen(v: torch.Tensor, v_gen: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v, v_gen.detach())


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------

def build_v_init(target_vel_np: np.ndarray, smooth_sigma: float, device: torch.device) -> torch.Tensor:
    v_np  = v_normalize_np(target_vel_np)
    mu_np = gaussian_filter(v_np, sigma=smooth_sigma).astype(np.float32)
    mu    = torch.from_numpy(mu_np).reshape(1, 1, 70, 70).to(device)
    return mu.clamp(-1.0, 1.0).requires_grad_(True)


# ---------------------------------------------------------------------------
# lambda warmup schedule
# ---------------------------------------------------------------------------

def compute_lambda(step: int, warmup_steps: int, ramp_steps: int, lambda_max: float) -> float:
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return lambda_max * progress


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Joint Alternating Optimization: v + z  (FlatFault-B)")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_joint_opt.yaml"),
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
    dcfg       = cfg.get("ddim") or {}
    model_path = _resolve_path(dcfg["model_path"])
    sched_cfg  = dcfg.get("scheduler_config") or {}

    wrapper, ddpm_sch, ddim_sch = load_model(model_path, sched_cfg)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)

    num_steps       = int(dcfg.get("num_steps", 20))
    num_steps_final = int(dcfg.get("num_steps_final", 20))
    eta             = float(dcfg.get("eta", 0.0))

    ddim_sch.set_timesteps(num_steps)
    t_start             = int(ddim_sch.timesteps[0].item())
    alphas_cumprod      = ddpm_sch.alphas_cumprod
    final_alpha_cumprod = float(ddim_sch.final_alpha_cumprod)

    # --- data ---
    ff_cfg       = cfg.get("flatfault") or {}
    data_path    = _resolve_path(ff_cfg["data_path"])
    sample_index = int(ff_cfg.get("sample_index", 0))
    if not data_path.is_file():
        raise SystemExit(f"Data not found: {data_path}")

    target_vel_np = np.load(data_path)[sample_index, 0].astype(np.float32)
    data_label    = f"{data_path.name}[{sample_index}]"

    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # --- initialize v and z ---
    init_cfg     = cfg.get("initialization") or {}
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))

    v = build_v_init(target_vel_np, smooth_sigma, device)

    z = torch.randn(1, 1, 70, 70, device=device, dtype=torch.float32,
                    generator=torch.Generator(device=device).manual_seed(seed))
    z = z.requires_grad_(True)

    # --- optimization config ---
    opt_cfg          = cfg.get("optimization") or {}
    total_steps      = int(opt_cfg.get("total_steps", 600))
    lr_v             = float(opt_cfg.get("lr_v", 0.03))
    lr_z             = float(opt_cfg.get("lr_z", 0.02))
    z_steps_per_iter = int(opt_cfg.get("z_steps_per_iter", 1))
    snapshots        = int(opt_cfg.get("snapshots", 8))
    wave_loss_type   = str(opt_cfg.get("wave_loss", "l1")).lower()

    guide_cfg    = cfg.get("guidance") or {}
    warmup_steps = int(guide_cfg.get("warmup_steps", 150))
    ramp_steps   = int(guide_cfg.get("ramp_steps", 100))
    lambda_max   = float(guide_cfg.get("lambda_max", 0.75))

    viz            = cfg.get("visualization") or {}
    vel_vmin       = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax       = float(viz.get("vel_vmax_m_s", 4500.0))
    wave_plot_shot = int(viz.get("wave_plot_shot", 0))

    # --- output directory ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_joint_opt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg)
    cfg_dump["_resolved"] = {
        "model_path": str(model_path), "data_path": str(data_path),
        "output_dir": str(out_dir), "device_used": str(device),
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dump, f, allow_unicode=True, sort_keys=False)

    print(
        f"[Joint Opt | FlatFault-B]  model={model_path.name}  data={data_label}  "
        f"total={total_steps}  lr_v={lr_v}  lr_z={lr_z}  "
        f"warmup={warmup_steps}  ramp={ramp_steps}  λ_max={lambda_max}  "
        f"num_steps={num_steps}  t_start={t_start}  z_steps={z_steps_per_iter}  device={device}"
    )

    # --- optimizers ---
    opt_v   = torch.optim.Adam([v], lr=lr_v)
    sched_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=total_steps, eta_min=0.0)
    opt_z   = torch.optim.Adam([z], lr=lr_z)
    sched_z = torch.optim.lr_scheduler.CosineAnnealingLR(opt_z, T_max=total_steps, eta_min=0.0)

    snap_times = set(
        int(x) for x in np.clip(
            np.round(np.linspace(0, total_steps, snapshots, endpoint=True)).astype(int),
            0, total_steps,
        )
    )

    hist_v_vel:       list[np.ndarray] = []
    hist_vgen_vel:    list[np.ndarray] = []
    hist_z_arr:       list[np.ndarray] = []
    hist_labels:      list[str]        = []
    hist_snap_mae_v:  list[float]      = []
    hist_snap_ssim_v: list[float]      = []
    hist_snap_mae_z:  list[float]      = []
    hist_snap_ssim_z: list[float]      = []

    wave_losses:  list[float] = []
    guide_losses: list[float] = []
    z_losses:     list[float] = []
    lambdas:      list[float] = []
    mae_v_hist:   list[float] = []
    ssim_v_hist:  list[float] = []
    mae_z_hist:   list[float] = []
    ssim_z_hist:  list[float] = []

    def capture(label: str, v_gen_np: np.ndarray,
                mae_v: float, ssim_v: float,
                mae_z: float, ssim_z: float) -> None:
        hist_v_vel.append(v_denormalize_np(v).copy())
        hist_vgen_vel.append(v_gen_np.copy())
        hist_z_arr.append(z.detach().cpu().numpy().squeeze().copy())
        hist_labels.append(label)
        hist_snap_mae_v.append(mae_v)
        hist_snap_ssim_v.append(ssim_v)
        hist_snap_mae_z.append(mae_z)
        hist_snap_ssim_z.append(ssim_z)

    with torch.no_grad():
        v_gen_init = sample_no_grad(wrapper, ddim_sch, z, num_steps, eta)
    if 0 in snap_times:
        vel_v0  = v_denormalize_np(v)
        vel_z0  = v_denormalize_np(v_gen_init)
        mv0, sv0 = velocity_mae_ssim(vel_v0, target_vel_np, device, vel_vmin, vel_vmax)
        mz0, sz0 = velocity_mae_ssim(vel_z0, target_vel_np, device, vel_vmin, vel_vmax)
        capture("iter 0 (init)", vel_z0, mv0, sv0, mz0, sz0)

    # =========================================================================
    # Main alternating optimization loop
    # =========================================================================
    for it in range(total_steps):

        # ── Step 1: Update z ─────────────────────────────────────────────────
        loss_z_val = 0.0
        for _ in range(z_steps_per_iter):
            opt_z.zero_grad(set_to_none=True)
            v_gen_z = sample_with_grad(wrapper, ddim_sch, z, num_steps, eta)
            loss_z  = F.mse_loss(v_gen_z, v.detach())
            loss_z.backward()
            opt_z.step()
            loss_z_val = loss_z.item()
        sched_z.step()

        # ── Step 2: Decode v_gen ──────────────────────────────────────────────
        v_gen = sample_no_grad(wrapper, ddim_sch, z, num_steps, eta)

        # ── Step 3: Update v ──────────────────────────────────────────────────
        lam = compute_lambda(it, warmup_steps, ramp_steps, lambda_max)
        opt_v.zero_grad(set_to_none=True)

        wave_pred = forward_wave(v)
        if wave_loss_type == "l1":
            loss_wave = F.l1_loss(wave_pred, target_wave)
        else:
            loss_wave = F.mse_loss(wave_pred, target_wave)

        if lam > 0.0:
            loss_guide     = guidance_from_vgen(v, v_gen)
            loss_v         = loss_wave + lam * loss_guide
            guide_loss_val = loss_guide.item()
        else:
            loss_v         = loss_wave
            guide_loss_val = 0.0

        loss_v.backward()
        opt_v.step()
        sched_v.step()

        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        wave_losses.append(loss_wave.item())
        guide_losses.append(guide_loss_val)
        z_losses.append(loss_z_val)
        lambdas.append(lam)

        with torch.no_grad():
            vel_v_np = v_denormalize_np(v)
            vel_z_np = v_denormalize_np(v_gen)
            mae_v, ssim_v = velocity_mae_ssim(vel_v_np, target_vel_np, device, vel_vmin, vel_vmax)
            mae_z, ssim_z = velocity_mae_ssim(vel_z_np, target_vel_np, device, vel_vmin, vel_vmax)
        mae_v_hist.append(mae_v)
        ssim_v_hist.append(ssim_v)
        mae_z_hist.append(mae_z)
        ssim_z_hist.append(ssim_z)

        if (it + 1) in snap_times:
            capture(f"iter {it + 1}", vel_z_np, mae_v, ssim_v, mae_z, ssim_z)

        if (it + 1) % max(1, total_steps // 10) == 0 or it == 0:
            print(
                f"  iter {it+1:4d}/{total_steps}  "
                f"wave={loss_wave.item():.5f}  guide={guide_loss_val:.5f}  "
                f"z_mse={loss_z_val:.5f}  λ={lam:.3f}  "
                f"MAE_v={mae_v:.1f}  SSIM_v={ssim_v:.4f}  "
                f"MAE_z={mae_z:.1f}  SSIM_z={ssim_z:.4f}"
            )

    # =========================================================================
    # Figures
    # =========================================================================

    # --- Figure 1: evolution snapshots ---
    ncols     = 1 + len(hist_v_vel)
    nrows     = 4
    row_titles = ["v  (physical)", "v_gen  (manifold)", "v − v_gen", "z  (latent noise)"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes[0, 0].set_title(f"target v\n{data_label}", fontsize=8)
    axes[0, 0].axis("off")
    for row_idx in range(1, nrows):
        axes[row_idx, 0].text(
            0.5, 0.5, row_titles[row_idx],
            transform=axes[row_idx, 0].transAxes,
            fontsize=8, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eeeeee", edgecolor="gray", alpha=0.9),
        )
        axes[row_idx, 0].axis("off")
    axes[0, 0].set_ylabel(row_titles[0], fontsize=8)

    all_z = np.concatenate([zj.ravel() for zj in hist_z_arr])
    zlim  = max(float(np.abs(all_z).max()), 3.0)

    for j, (vv, vg, zj, lbl, mv, sv, mz, sz) in enumerate(zip(
        hist_v_vel, hist_vgen_vel, hist_z_arr, hist_labels,
        hist_snap_mae_v, hist_snap_ssim_v, hist_snap_mae_z, hist_snap_ssim_z,
    )):
        col = j + 1
        axes[0, col].imshow(vv, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, col].set_title(f"{lbl}\nMAE={mv:.0f} m/s  SSIM={sv:.3f}", fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(vg, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[1, col].set_title(f"MAE={mz:.0f} m/s  SSIM={sz:.3f}", fontsize=7)
        axes[1, col].axis("off")
        diff = vv - vg
        dlim = max(float(np.abs(diff).max()), 1.0)
        axes[2, col].imshow(diff, cmap="coolwarm", aspect="auto", vmin=-dlim, vmax=dlim)
        axes[2, col].axis("off")
        axes[3, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-zlim, vmax=zlim)
        axes[3, col].axis("off")

    plt.suptitle(
        f"FlatFault-B Joint Opt evolution  |  warmup={warmup_steps}  λ_max={lambda_max}  {data_label}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 2: final result comparison ---
    final_v_np    = v_denormalize_np(v)
    final_vgen_np = v_denormalize_np(sample_no_grad(wrapper, ddim_sch, z, num_steps, eta))
    mae_v_final,  ssim_v_final  = velocity_mae_ssim(final_v_np,    target_vel_np, device, vel_vmin, vel_vmax)
    mae_vg_final, ssim_vg_final = velocity_mae_ssim(final_vgen_np, target_vel_np, device, vel_vmin, vel_vmax)

    with torch.no_grad():
        wave_final = forward_wave(v)

    err_v  = final_v_np   - target_vel_np
    err_vg = final_vgen_np - target_vel_np
    elim_v  = max(float(np.abs(err_v).max()),  1e-6)
    elim_vg = max(float(np.abs(err_vg).max()), 1e-6)
    sh  = wave_plot_shot
    tw  = target_wave[sh].detach().cpu().numpy()
    pw  = wave_final[sh].detach().cpu().numpy()
    ew  = pw - tw
    wlim = max(np.abs(tw).max(), np.abs(pw).max()) + 1e-6
    elw  = float(np.abs(ew).max()) + 1e-6

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[0], [
        (target_vel_np, "target v",      "viridis",  vel_vmin,  vel_vmax),
        (final_v_np,    "v (physical)",  "viridis",  vel_vmin,  vel_vmax),
        (err_v,         "v error",       "coolwarm", -elim_v,   elim_v),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[1], [
        (target_vel_np,  "target v",       "viridis",  vel_vmin,  vel_vmax),
        (final_vgen_np,  "v_gen (latent)", "viridis",  vel_vmin,  vel_vmax),
        (err_vg,         "v_gen error",    "coolwarm", -elim_vg,  elim_vg),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes[2], [
        (tw.T, f"target wave shot {sh}", "seismic", -wlim, wlim),
        (pw.T, f"pred wave shot {sh}",   "seismic", -wlim, wlim),
        (ew.T, "wave residual",          "seismic", -elw,  elw),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        f"FlatFault-B Joint Opt  |  warmup={warmup_steps}  λ_max={lambda_max}  {data_label}\n"
        f"v:     MAE={mae_v_final:.1f} m/s  SSIM={ssim_v_final:.4f}\n"
        f"v_gen: MAE={mae_vg_final:.1f} m/s  SSIM={ssim_vg_final:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 3: metrics ---
    iters = list(range(1, total_steps + 1))
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes[0, 0].plot(iters, wave_losses,  color="C0"); axes[0, 0].set_title("Wave loss (v)");          axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(iters, guide_losses, color="C5"); axes[0, 1].set_title("||v − v_gen||² (guide)"); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(iters, lambdas,      color="C4"); axes[0, 2].set_title("λ schedule");             axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(iters, z_losses,     color="C3"); axes[1, 0].set_title("z MSE loss");             axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(iters, mae_v_hist,   color="C1", label="v");    axes[1, 1].plot(iters, mae_z_hist, color="C2", label="v_gen", linestyle="--")
    axes[1, 1].set_title("MAE (m/s)"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].plot(iters, ssim_v_hist,  color="C1", label="v");    axes[1, 2].plot(iters, ssim_z_hist, color="C2", label="v_gen", linestyle="--")
    axes[1, 2].set_title("SSIM"); axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.3)
    axes[2, 0].plot(iters, [m - mz for m, mz in zip(mae_v_hist, mae_z_hist)], color="C6")
    axes[2, 0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[2, 0].set_title("MAE_v − MAE_z  (>0: z closer to truth)"); axes[2, 0].set_xlabel("Iteration"); axes[2, 0].grid(True, alpha=0.3)
    axes[2, 1].plot(iters, [s - sz for s, sz in zip(ssim_v_hist, ssim_z_hist)], color="C7")
    axes[2, 1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[2, 1].set_title("SSIM_v − SSIM_z  (<0: z better)"); axes[2, 1].set_xlabel("Iteration"); axes[2, 1].grid(True, alpha=0.3)
    axes[2, 2].axis("off")
    plt.suptitle(
        f"FlatFault-B Joint Opt metrics  |  warmup={warmup_steps}  ramp={ramp_steps}  λ_max={lambda_max}  "
        f"{data_label}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 4: final generation from optimized z ---
    GEN_STEPS = num_steps_final
    with torch.no_grad():
        v_final_gen = sample_custom_no_grad(
            wrapper, alphas_cumprod, z, t_start, GEN_STEPS, eta, final_alpha_cumprod
        )
    v_final_gen_np = v_denormalize_np(v_final_gen)
    mae_gen, ssim_gen = velocity_mae_ssim(v_final_gen_np, target_vel_np, device, vel_vmin, vel_vmax)
    err_gen  = v_final_gen_np - target_vel_np
    elim_gen = max(float(np.abs(err_gen).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (arr, title, cmap, vm0, vm1) in zip(axes, [
        (target_vel_np,  "target v",  "viridis",  vel_vmin, vel_vmax),
        (v_final_gen_np, f"DDIM({GEN_STEPS} steps) from z_opt\nMAE={mae_gen:.1f} m/s  SSIM={ssim_gen:.4f}",
                         "viridis",  vel_vmin, vel_vmax),
        (err_gen,        "error (m/s)",  "coolwarm", -elim_gen, elim_gen),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(
        f"FlatFault-B Final generation  |  z_opt → DDIM({GEN_STEPS} steps)  |  {data_label}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "final_generation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  z→DDIM({GEN_STEPS}) MAE={mae_gen:.1f} m/s  SSIM={ssim_gen:.4f}")

    # =========================================================================
    # Phase 2: wave-domain latent optimization
    # =========================================================================
    p2_cfg    = cfg.get("phase2") or {}
    p2_steps  = int(p2_cfg.get("opt_steps", 100))
    p2_lr     = float(p2_cfg.get("lr", 0.01))
    p2_nsteps = int(p2_cfg.get("num_steps", num_steps))
    p2_snaps  = int(p2_cfg.get("snapshots", 6))

    print(f"\n[Phase 2]  wave-domain latent opt  steps={p2_steps}  lr={p2_lr}  "
          f"ddim_steps={p2_nsteps}  t_start={t_start}")

    opt_z2   = torch.optim.Adam([z], lr=p2_lr)
    sched_z2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_z2, T_max=max(1, p2_steps), eta_min=0.0)

    p2_snap_times = set(
        int(x) for x in np.clip(
            np.round(np.linspace(0, p2_steps, p2_snaps, endpoint=True)).astype(int),
            0, p2_steps,
        )
    )

    p2_hist_vel:    list[np.ndarray] = []
    p2_hist_z:      list[np.ndarray] = []
    p2_hist_labels: list[str]        = []
    p2_snap_mae:    list[float]      = []
    p2_snap_ssim:   list[float]      = []
    p2_wave_losses: list[float]      = []
    p2_mae_hist:    list[float]      = []
    p2_ssim_hist:   list[float]      = []

    if 0 in p2_snap_times:
        v_p2_init_np = v_denormalize_np(
            sample_custom_no_grad(wrapper, alphas_cumprod, z, t_start, p2_nsteps, eta, final_alpha_cumprod)
        )
        m0, s0 = velocity_mae_ssim(v_p2_init_np, target_vel_np, device, vel_vmin, vel_vmax)
        p2_hist_vel.append(v_p2_init_np.copy())
        p2_hist_z.append(z.detach().cpu().numpy().squeeze().copy())
        p2_hist_labels.append("P2 iter 0")
        p2_snap_mae.append(m0)
        p2_snap_ssim.append(s0)

    for it in range(p2_steps):
        opt_z2.zero_grad(set_to_none=True)
        pred_n    = sample_custom_with_grad(
            wrapper, alphas_cumprod, z, t_start, p2_nsteps, eta, final_alpha_cumprod
        )
        wave_pred = forward_wave(pred_n)
        loss_p2   = F.mse_loss(wave_pred, target_wave)
        loss_p2.backward()
        opt_z2.step()
        sched_z2.step()

        with torch.no_grad():
            vel_p2 = v_denormalize_np(pred_n.detach())
            mae_p2, ssim_p2 = velocity_mae_ssim(vel_p2, target_vel_np, device, vel_vmin, vel_vmax)
        p2_wave_losses.append(loss_p2.item())
        p2_mae_hist.append(mae_p2)
        p2_ssim_hist.append(ssim_p2)

        if (it + 1) in p2_snap_times:
            p2_hist_vel.append(vel_p2.copy())
            p2_hist_z.append(z.detach().cpu().numpy().squeeze().copy())
            p2_hist_labels.append(f"P2 iter {it + 1}")
            p2_snap_mae.append(mae_p2)
            p2_snap_ssim.append(ssim_p2)

        if (it + 1) % max(1, p2_steps // 5) == 0 or it == 0:
            print(
                f"  [P2] iter {it+1:4d}/{p2_steps}  "
                f"wave_mse={loss_p2.item():.5f}  MAE={mae_p2:.1f}  SSIM={ssim_p2:.4f}"
            )

    # --- Figure 5: Phase 2 evolution ---
    p2_ncols      = 1 + len(p2_hist_vel)
    p2_nrows      = 3
    p2_row_titles = ["v_gen  (DDIM(z))", "v_gen − target  error", "z  (latent noise)"]
    fig5, ax5 = plt.subplots(p2_nrows, p2_ncols, figsize=(2.6 * p2_ncols, 2.8 * p2_nrows))
    ax5 = np.atleast_2d(ax5).reshape(p2_nrows, p2_ncols)

    ax5[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    ax5[0, 0].set_title(f"target v\n{data_label}", fontsize=8); ax5[0, 0].axis("off")
    for row_idx in range(1, p2_nrows):
        ax5[row_idx, 0].text(0.5, 0.5, p2_row_titles[row_idx],
            transform=ax5[row_idx, 0].transAxes, fontsize=8, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eeeeee", edgecolor="gray", alpha=0.9))
        ax5[row_idx, 0].axis("off")
    ax5[0, 0].set_ylabel(p2_row_titles[0], fontsize=8)

    p2_all_z = np.concatenate([zj.ravel() for zj in p2_hist_z]) if p2_hist_z else np.zeros(1)
    p2_zlim  = max(float(np.abs(p2_all_z).max()), 3.0)

    for j, (vel, zj, lbl, mv, sv) in enumerate(zip(
        p2_hist_vel, p2_hist_z, p2_hist_labels, p2_snap_mae, p2_snap_ssim,
    )):
        col = j + 1
        ax5[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        ax5[0, col].set_title(f"{lbl}\nMAE={mv:.0f} m/s  SSIM={sv:.3f}", fontsize=7)
        ax5[0, col].axis("off")
        err_p2  = vel - target_vel_np
        elim_p2 = max(float(np.abs(err_p2).max()), 1.0)
        ax5[1, col].imshow(err_p2, cmap="coolwarm", aspect="auto", vmin=-elim_p2, vmax=elim_p2)
        ax5[1, col].axis("off")
        ax5[2, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-p2_zlim, vmax=p2_zlim)
        ax5[2, col].axis("off")

    plt.suptitle(
        f"FlatFault-B Phase 2 evolution  |  wave-domain latent opt  |  {data_label}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 6: Phase 2 final generation ---
    with torch.no_grad():
        v_p2_final_gen = sample_custom_no_grad(
            wrapper, alphas_cumprod, z, t_start, num_steps_final, eta, final_alpha_cumprod
        )
    v_p2_final_np  = v_denormalize_np(v_p2_final_gen)
    mae_p2_final, ssim_p2_final = velocity_mae_ssim(v_p2_final_np, target_vel_np, device, vel_vmin, vel_vmax)
    err_p2f  = v_p2_final_np - target_vel_np
    elim_p2f = max(float(np.abs(err_p2f).max()), 1e-6)

    fig6, ax6 = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (arr, title, cmap, vm0, vm1) in zip(ax6, [
        (target_vel_np, "target v", "viridis", vel_vmin, vel_vmax),
        (v_p2_final_np,
         f"DDIM({num_steps_final} steps) from z_phase2\nMAE={mae_p2_final:.1f} m/s  SSIM={ssim_p2_final:.4f}",
         "viridis", vel_vmin, vel_vmax),
        (err_p2f, "error (m/s)", "coolwarm", -elim_p2f, elim_p2f),
    ]):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
        ax.set_title(title, fontsize=9); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(
        f"FlatFault-B Phase 2 final generation  |  z_phase2 → DDIM({num_steps_final} steps)  |  {data_label}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_final_generation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [P2] z→DDIM({num_steps_final}) MAE={mae_p2_final:.1f} m/s  SSIM={ssim_p2_final:.4f}")

    # --- Figure 7: Phase 2 metrics ---
    if p2_wave_losses:
        p2_iters = list(range(1, p2_steps + 1))
        fig7, ax7 = plt.subplots(1, 3, figsize=(14, 4))
        ax7[0].plot(p2_iters, p2_wave_losses, color="C0")
        ax7[0].set_title("P2 Wave MSE loss"); ax7[0].set_xlabel("Iteration"); ax7[0].grid(True, alpha=0.3)
        ax7[1].plot(p2_iters, p2_mae_hist, color="C1")
        ax7[1].set_title("P2 MAE (m/s)"); ax7[1].set_xlabel("Iteration"); ax7[1].grid(True, alpha=0.3)
        ax7[2].plot(p2_iters, p2_ssim_hist, color="C2")
        ax7[2].set_title("P2 SSIM"); ax7[2].set_xlabel("Iteration"); ax7[2].grid(True, alpha=0.3)
        plt.suptitle(f"FlatFault-B Phase 2 metrics  |  {data_label}", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "phase2_metrics.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- summary ---
    summary = {
        "dataset": "FlatFault-B",
        "data_file": str(data_path),
        "sample_index": sample_index,
        "init_smooth_sigma": smooth_sigma,
        "warmup_steps": warmup_steps, "ramp_steps": ramp_steps, "lambda_max": lambda_max,
        "final_wave_loss": float(wave_losses[-1]),
        "final_guide_loss": float(guide_losses[-1]),
        "final_z_mse_loss": float(z_losses[-1]),
        "final_v_mae_m_s": float(mae_v_hist[-1]),
        "final_v_ssim": float(ssim_v_hist[-1]),
        "final_vgen_mae_m_s": float(mae_z_hist[-1]),
        "final_vgen_ssim": float(ssim_z_hist[-1]),
        f"final_gen_{GEN_STEPS}steps_mae_m_s": float(mae_gen),
        f"final_gen_{GEN_STEPS}steps_ssim": float(ssim_gen),
        "phase2_opt_steps": p2_steps,
        f"phase2_gen_{num_steps_final}steps_mae_m_s": float(mae_p2_final),
        f"phase2_gen_{num_steps_final}steps_ssim": float(ssim_p2_final),
        "phase2_final_wave_mse": float(p2_wave_losses[-1]) if p2_wave_losses else None,
        "phase2_final_mae_m_s": float(p2_mae_hist[-1]) if p2_mae_hist else None,
        "phase2_final_ssim": float(p2_ssim_hist[-1]) if p2_ssim_hist else None,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {out_dir}")
    print(f"  [P1] v    MAE={mae_v_hist[-1]:.1f} m/s  SSIM={ssim_v_hist[-1]:.4f}")
    print(f"  [P1] vgen MAE={mae_z_hist[-1]:.1f} m/s  SSIM={ssim_z_hist[-1]:.4f}")
    print(f"  [P2] final MAE={mae_p2_final:.1f} m/s  SSIM={ssim_p2_final:.4f}")


if __name__ == "__main__":
    main()
