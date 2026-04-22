#!/usr/bin/env python3
"""
DDIM Stochastic Sample Inversion (SSI).

For each of n_samples, the following flow is executed:

  1. Load clean normalized velocity x_0 ∈ [-1, 1]  (shape 1×1×70×70)

  2. Jump to discrete timestep t_j by sampling from the DDPM marginal:
       x_{t_j} = sqrt(ᾱ_{t_j}) · x_0 + sqrt(1 - ᾱ_{t_j}) · εᵢ,  εᵢ ~ N(0,I)
     Different εᵢ → different x_{t_j} for the same x_0.

  3. DDIM inversion (ODE run forward in time, η=0):
     Starting from x_{t_j}, iterate through the DDIM timestep list in
     ascending order (t_j → t_max) using the formula:
       x̂_0 = (x_t - sqrt(1-ᾱ_t) · ε_θ) / sqrt(ᾱ_t)
       x_{t'} = sqrt(ᾱ_{t'}) · x̂_0 + sqrt(1-ᾱ_{t'}) · ε_θ   (t' > t)
     The result x_{t_max} ≈ N(0, I) is the inverted noise zᵢ.

  4. DDIM sampling from zᵢ → x̂_0ᵢ  (standard reverse ODE, η=0).

  5. Compute MAE(x_0_physical, x̂_0ᵢ_physical) in m/s.

Outputs:
  ssi_noises.png        — 3×3 grid of inverted noises zᵢ
  ssi_reconstructions.png — target + 3×3 grid of reconstructed velocities

Run (from Manifold_constrained_FWI/):
  uv run python exp/DDIM/ddim_ssi/demo_ddim_ssi.py \\
    --config exp/DDIM/ddim_ssi/config_ddim_ssi.yaml
"""

from __future__ import annotations

import argparse
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
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR = _MANIFOLD_ROOT / "training"
_CURVE_VEL_B = _MANIFOLD_ROOT.parent / "CurveVelB"

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def v_denormalize(v_norm: torch.Tensor | np.ndarray) -> np.ndarray:
    """[-1, 1] → physical velocity in m/s."""
    if isinstance(v_norm, torch.Tensor):
        v_norm = v_norm.detach().float().cpu().numpy()
    return v_norm.astype(np.float32) * 1500.0 + 3000.0


def load_ddim_pipeline(checkpoint: Path, training_yaml: Path | None):
    checkpoint = checkpoint.resolve()
    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}")

    torch_dtype = torch.float32
    if training_yaml is not None and training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s = str((cfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        torch_dtype = torch.float16 if s in ("float16", "fp16") else torch.float32

    wrapper = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)
    ddpm_sched = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    ddim_sched = DDIMScheduler.from_config(ddpm_sched.config, clip_sample=False)
    return wrapper, ddim_sched


# ---------------------------------------------------------------------------
# DDIM sampling (standard reverse ODE)
# ---------------------------------------------------------------------------

def ddim_sample(wrapper, ddim_sched, z: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Standard DDIM sampling (η=0). z: (1,1,70,70) → (1,1,70,70)."""
    ddim_sched.set_timesteps(num_steps)
    image = z
    with torch.no_grad():
        for t in ddim_sched.timesteps:
            model_output = wrapper(image, t).sample
            image = ddim_sched.step(
                model_output, t, image, eta=0.0, use_clipped_model_output=False
            ).prev_sample
    return image  # (1, 1, 70, 70)


# ---------------------------------------------------------------------------
# DDIM inversion (forward ODE, ascending timesteps)
# ---------------------------------------------------------------------------

def ddim_inversion(
    wrapper,
    ddim_sched,
    x_tj: torch.Tensor,
    jump_step_idx: int,
) -> torch.Tensor:
    """Run DDIM inversion from x_{t_j} to x_{t_max}.

    Uses the DDIM ODE in the forward direction (ascending timesteps).
    Formula (η=0):
        ε_θ  = model(x_t, t)
        x̂_0  = (x_t  - sqrt(1-ᾱ_t) · ε_θ) / sqrt(ᾱ_t)
        x_{t'} = sqrt(ᾱ_{t'}) · x̂_0 + sqrt(1-ᾱ_{t'}) · ε_θ   (t' > t)

    Args:
        x_tj:            Noisy sample at timestep t_j, shape (1,1,70,70).
        jump_step_idx:   Index into the ascending inversion timestep list
                         (inversion_ts = ddim_sched.timesteps.flip(0)).
                         x_tj lives at inversion_ts[jump_step_idx].

    Returns:
        x_{t_max}: inverted noise, shape (1,1,70,70), approximately N(0,I).
    """
    # Ascending timestep list (inversion direction)
    inversion_ts = ddim_sched.timesteps.flip(0)  # [t_K, ..., t_1], t_1 = t_max
    n_inv = len(inversion_ts)

    image = x_tj.clone()
    with torch.no_grad():
        # Step from index jump_step_idx up to n_inv-2 (inclusive).
        # Each step maps x_{inversion_ts[i]} → x_{inversion_ts[i+1]}.
        for i in range(jump_step_idx, n_inv - 1):
            t_cur  = inversion_ts[i]
            t_next = inversion_ts[i + 1]

            alpha_t      = ddim_sched.alphas_cumprod[t_cur].to(image.device)
            alpha_t_next = ddim_sched.alphas_cumprod[t_next].to(image.device)

            eps_theta = wrapper(image, t_cur).sample          # predicted noise
            x0_pred   = (image - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()
            image     = alpha_t_next.sqrt() * x0_pred + (1 - alpha_t_next).sqrt() * eps_theta

    return image  # (1, 1, 70, 70)


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM SSI experiment")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_ssi.yaml"),
    )
    args = parser.parse_args()
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

    inv_steps    = int(dcfg.get("inv_steps", dcfg.get("num_steps", 20)))
    sample_steps = int(dcfg.get("sample_steps", dcfg.get("num_steps", 20)))

    # Two separate schedulers so set_timesteps calls don't interfere
    sample_sched = DDIMScheduler.from_config(inv_sched.config, clip_sample=False)

    # Set up inversion timestep grid
    inv_sched.set_timesteps(inv_steps)

    # Ascending inversion timestep list and the jump point
    inversion_ts   = inv_sched.timesteps.flip(0)  # [t_K, ..., t_1]
    ssi_cfg        = cfg.get("ssi") or {}
    n_samples      = int(ssi_cfg.get("n_samples", 9))
    jump_step_idx  = int(ssi_cfg.get("jump_step_idx", inv_steps // 2))
    seed_start     = int(ssi_cfg.get("seed_start", 0))

    jump_step_idx = min(jump_step_idx, inv_steps - 2)  # leave at least one inversion step
    t_j           = inversion_ts[jump_step_idx].item()
    alpha_bar_tj  = inv_sched.alphas_cumprod[t_j].to(device)

    # --- target velocity ---
    cv = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv["model60_path"])
    sample_index = int(cv.get("sample_index", 0))
    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    raw = np.load(model60_path)[sample_index, 0].astype(np.float32)   # (70, 70) m/s
    x0_norm = torch.from_numpy(
        (raw / 1500.0 - 2.0).astype(np.float32)                       # [-1,1]
    ).reshape(1, 1, 70, 70).to(device)
    x0_phys = v_denormalize(x0_norm).squeeze()  # numpy (70,70)

    viz     = cfg.get("visualization") or {}
    vel_vmin = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax = float(viz.get("vel_vmax_m_s", 4500.0))

    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_ssi_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[DDIM SSI]  checkpoint={ckpt.name}  inv_steps={inv_steps}  sample_steps={sample_steps}  "
        f"jump_step_idx={jump_step_idx}/{inv_steps-1}  t_j={t_j}  "
        f"alpha_bar={alpha_bar_tj.item():.4f}  n_samples={n_samples}  device={device}"
    )
    print(f"  sqrt(ᾱ_tj)={alpha_bar_tj.sqrt().item():.4f}  "
          f"sqrt(1-ᾱ_tj)={(1-alpha_bar_tj).sqrt().item():.4f}")

    # --- SSI loop ---
    inverted_noises: list[np.ndarray] = []  # each (70,70)
    recon_vels:      list[np.ndarray] = []  # each (70,70) in m/s
    mae_list:        list[float]      = []

    for i in range(n_samples):
        g = torch.Generator(device=device).manual_seed(seed_start + i)
        eps = torch.randn(1, 1, 70, 70, device=device, generator=g)

        # Step 2: jump to t_j
        x_tj = alpha_bar_tj.sqrt() * x0_norm + (1 - alpha_bar_tj).sqrt() * eps

        # Step 3: DDIM inversion from x_{t_j} to x_{t_max}
        z_i = ddim_inversion(wrapper, inv_sched, x_tj, jump_step_idx)

        # Step 4: DDIM sampling from z_i
        x_hat = ddim_sample(wrapper, sample_sched, z_i, sample_steps)

        # Step 5: MAE in m/s
        x_hat_phys = v_denormalize(x_hat.squeeze())
        mae = float(np.mean(np.abs(x_hat_phys - x0_phys)))
        mae_list.append(mae)

        # record z_i statistics
        z_np = z_i.detach().cpu().numpy().squeeze()
        zmean, zstd = float(z_np.mean()), float(z_np.std())
        inverted_noises.append(z_np)
        recon_vels.append(x_hat_phys)

        print(f"  sample {i+1}/{n_samples}  t_j={t_j}  "
              f"z mean={zmean:+.3f}  z std={zstd:.3f}  MAE={mae:.1f} m/s")

    print(f"\nMAE  mean={np.mean(mae_list):.1f}  "
          f"min={np.min(mae_list):.1f}  max={np.max(mae_list):.1f} m/s")

    # -----------------------------------------------------------------------
    # Figure 1: 3×3 grid of inverted noises
    # -----------------------------------------------------------------------
    nrows_grid = int(np.ceil(n_samples / 3))
    all_z = np.concatenate([z.ravel() for z in inverted_noises])
    zlim = max(3.0, float(np.percentile(np.abs(all_z), 99)) + 0.1)

    fig, axes = plt.subplots(nrows_grid, 3, figsize=(8.5, 2.8 * nrows_grid))
    axes_flat = np.array(axes).ravel()
    for idx, (z_np, ax) in enumerate(zip(inverted_noises, axes_flat)):
        zmean = float(z_np.mean())
        zstd  = float(z_np.std())
        ax.imshow(z_np, cmap="coolwarm", aspect="auto", vmin=-zlim, vmax=zlim)
        ax.set_title(f"z_{idx+1}\nμ={zmean:+.3f}  σ={zstd:.3f}", fontsize=8)
        ax.axis("off")
    for ax in axes_flat[n_samples:]:
        ax.axis("off")
    plt.suptitle(
        f"Inverted noises  |  t_j={t_j}  (step {jump_step_idx}/{inv_steps-1})  "
        f"ᾱ_tj={alpha_bar_tj.item():.4f}\n"
        f"model60[{sample_index}]  checkpoint={ckpt.name}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "ssi_noises.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Figure 2: reference + 3×3 grid of reconstructed velocities
    # -----------------------------------------------------------------------
    n_cols = 4  # 1 reference + 3 per row
    n_plot_rows = nrows_grid
    fig, axes = plt.subplots(n_plot_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_plot_rows))
    axes = np.atleast_2d(axes).reshape(n_plot_rows, n_cols)

    # Column 0: target velocity (repeated for each row)
    for r in range(n_plot_rows):
        im = axes[r, 0].imshow(x0_phys, cmap="viridis", aspect="auto",
                               vmin=vel_vmin, vmax=vel_vmax)
        axes[r, 0].set_title("target v (m/s)" if r == 0 else "", fontsize=8)
        axes[r, 0].axis("off")
        if r == n_plot_rows - 1:
            plt.colorbar(im, ax=axes[r, 0], fraction=0.046)

    # Columns 1-3: reconstructed velocities
    for idx, (vel, mae) in enumerate(zip(recon_vels, mae_list)):
        r, c = divmod(idx, 3)
        ax = axes[r, c + 1]
        im = ax.imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        ax.set_title(f"recon {idx+1}\nMAE={mae:.1f} m/s", fontsize=8)
        ax.axis("off")
    for idx in range(n_samples, n_plot_rows * 3):
        r, c = divmod(idx, 3)
        axes[r, c + 1].axis("off")

    plt.suptitle(
        f"SSI reconstructions  |  t_j={t_j}  inv={inv_steps}/sample={sample_steps}  "
        f"MAE mean={np.mean(mae_list):.1f} m/s\n"
        f"model60[{sample_index}]  checkpoint={ckpt.name}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "ssi_reconstructions.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Figure 3: cosine similarity matrix of inverted noises
    # -----------------------------------------------------------------------
    Z = np.stack([z.ravel() for z in inverted_noises])      # (n_samples, 70*70)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)        # (n_samples, 1)
    Z_normed = Z / np.clip(norms, 1e-8, None)
    cos_sim = Z_normed @ Z_normed.T                          # (n_samples, n_samples)

    labels = [f"z{i+1}" for i in range(n_samples)]
    fig, ax = plt.subplots(figsize=(0.7 * n_samples + 1.5, 0.7 * n_samples + 1.5))
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    ax.set_xticks(range(n_samples)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n_samples)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n_samples):
        for j in range(n_samples):
            ax.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(cos_sim[i, j]) < 0.6 else "white")
    ax.set_title(
        f"Cosine similarity of inverted noises\n"
        f"t_j={t_j}  (step {jump_step_idx}/{inv_steps-1})  ᾱ_tj={alpha_bar_tj.item():.4f}  "
        f"model60[{sample_index}]",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "ssi_cosine_sim.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
