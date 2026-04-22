#!/usr/bin/env python3
"""
DDIM generation quality test.

Generates ``n_samples`` velocity fields from random noise using the custom
DDIM sampler (``src.core.generate.ddim_sample``) and visualises them in a
grid.  Designed to test generation quality at different ``(t_start, num_steps)``
combinations.

t_start=999 corresponds to the maximum-noise (pure Gaussian) starting point.
t_start=666 matches the effective start of the standard ``set_timesteps(3)``
HuggingFace DDIM schedule.

Run (from Manifold_constrained_FWI/):
  uv run python exp/DDIM/ddim_generate_test/demo_ddim_generate.py \\
    --config exp/DDIM/ddim_generate_test/config_ddim_generate.yaml
"""

from __future__ import annotations

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_SCRIPT_DIR    = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR  = _MANIFOLD_ROOT / "training"
_CURVE_VEL_B   = _MANIFOLD_ROOT.parent / "CurveVelB"

for _p in [str(_CURVE_VEL_B), str(_TRAINING_DIR), str(_MANIFOLD_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_null_fd      = os.open(os.devnull, os.O_WRONLY)
_saved_fd2    = os.dup(2)
_saved_stderr = sys.stderr
os.dup2(_null_fd, 2); os.close(_null_fd)
sys.stderr = open(os.devnull, "w")
try:
    from diffusers_torch_compat import ensure_diffusers_custom_ops_safe
    ensure_diffusers_custom_ops_safe()
    from diffusers import DDIMScheduler, DDPMScheduler
    from openfwi_unet_wrapper import load_openfwi_checkpoint
finally:
    sys.stderr.close()
    sys.stderr = _saved_stderr
    os.dup2(_saved_fd2, 2); os.close(_saved_fd2)

from src.core.generate import ddim_build_timesteps, ddim_sample


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint: Path):
    """Return (wrapper, ddpm_sch, ddim_sch)."""
    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}")

    training_yaml = (checkpoint.parent / "config_used.yaml").resolve()
    torch_dtype   = torch.float32
    if training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            tcfg = yaml.safe_load(f)
        s = str((tcfg.get("unet") or {}).get("torch_dtype", "float32")).lower()
        if s in ("float16", "fp16"):
            torch_dtype = torch.float16

    wrapper  = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch_dtype)
    ddpm_sch = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    ddim_sch = DDIMScheduler.from_config(ddpm_sch.config, clip_sample=False)
    return wrapper, ddpm_sch, ddim_sch


def v_denorm_np(v: torch.Tensor) -> np.ndarray:
    """Normalised [-1, 1] → physical velocity m/s."""
    return (v.detach().float().cpu().numpy().squeeze() * 1500.0 + 3000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM generation quality test")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_ddim_generate.yaml"),
    )
    args     = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path.resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device = torch.device(cfg.get("device", "cuda:0"))
    seed   = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- model ---
    dcfg = cfg.get("ddim") or {}
    ckpt = _resolve(dcfg["checkpoint"])
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    wrapper, ddpm_sch, ddim_sch = load_model(ckpt)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)

    alphas_cumprod      = ddpm_sch.alphas_cumprod            # (T,)
    final_alpha_cumprod = float(ddim_sch.final_alpha_cumprod)

    t_start   = int(dcfg.get("t_start", 999))
    num_steps = int(dcfg.get("num_steps", 3))
    eta       = float(dcfg.get("eta", 0.0))

    timesteps = ddim_build_timesteps(t_start, num_steps)

    # --- generation config ---
    gen_cfg   = cfg.get("generation") or {}
    n_samples = int(gen_cfg.get("n_samples", 9))

    viz       = cfg.get("visualization") or {}
    vel_vmin  = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax  = float(viz.get("vel_vmax_m_s", 4500.0))

    # --- output dir ---
    out_base = Path((cfg.get("paths") or {}).get("outdir", "demo_output"))
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"ddim_gen_t{t_start}_n{num_steps}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[DDIM generate]  checkpoint={ckpt.name}  "
        f"t_start={t_start}  num_steps={num_steps}  eta={eta}  "
        f"timesteps={timesteps}  n_samples={n_samples}  device={device}"
    )

    # --- generate ---
    ncols  = math.ceil(math.sqrt(n_samples))
    nrows  = math.ceil(n_samples / ncols)

    gen_images: list[np.ndarray] = []
    for i in range(n_samples):
        z = torch.randn(
            1, 1, 70, 70,
            device=device, dtype=torch.float32,
            generator=torch.Generator(device=device).manual_seed(seed + i),
        )
        v_norm = ddim_sample(
            wrapper, alphas_cumprod, z,
            t_start=t_start, num_steps=num_steps,
            eta=eta, final_alpha_cumprod=final_alpha_cumprod,
            no_grad=True,
        )
        gen_images.append(v_denorm_np(v_norm))

    # --- visualise ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for idx, (ax, vel) in enumerate(zip(axes_flat, gen_images)):
        im = ax.imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        ax.set_title(f"sample {idx}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # hide unused axes
    for ax in axes_flat[len(gen_images):]:
        ax.axis("off")

    plt.suptitle(
        f"DDIM generation  |  t_start={t_start}  num_steps={num_steps}  eta={eta}\n"
        f"timesteps: {timesteps}",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = out_dir / "generated_grid.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  saved → {out_path}")
    print(f"  vel range: [{min(v.min() for v in gen_images):.0f}, "
          f"{max(v.max() for v in gen_images):.0f}] m/s")

    # -----------------------------------------------------------------------
    # Comparison: same seeds × different step counts
    # Rows = seeds, columns = step counts.  All start from the same t_start.
    # -----------------------------------------------------------------------
    cmp_cfg = cfg.get("comparison") or {}
    if cmp_cfg.get("enabled", False):
        step_counts   = [int(s) for s in cmp_cfg.get("step_counts", [3, 6, 10, 20, 100])]
        n_cmp_samples = int(cmp_cfg.get("n_samples", 3))

        print(f"\n[comparison]  t_start={t_start}  step_counts={step_counts}  "
              f"n_samples={n_cmp_samples}")

        # cmp_grid[seed_idx][step_idx] = velocity np array
        cmp_grid: list[list[np.ndarray]] = []
        for si in range(n_cmp_samples):
            z = torch.randn(
                1, 1, 70, 70,
                device=device, dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(seed + si),
            )
            row: list[np.ndarray] = []
            for ns in step_counts:
                v_norm = ddim_sample(
                    wrapper, alphas_cumprod, z,
                    t_start=t_start, num_steps=ns,
                    eta=eta, final_alpha_cumprod=final_alpha_cumprod,
                    no_grad=True,
                )
                row.append(v_denorm_np(v_norm))
                print(f"  seed {si}  steps={ns:4d}  "
                      f"range=[{row[-1].min():.0f}, {row[-1].max():.0f}] m/s")
            cmp_grid.append(row)

        ncols_cmp = len(step_counts)
        nrows_cmp = n_cmp_samples
        fig2, axes2 = plt.subplots(
            nrows_cmp, ncols_cmp,
            figsize=(3.0 * ncols_cmp, 2.9 * nrows_cmp),
        )
        axes2 = np.array(axes2).reshape(nrows_cmp, ncols_cmp)

        for si in range(nrows_cmp):
            for ci, (ns, vel) in enumerate(zip(step_counts, cmp_grid[si])):
                ax = axes2[si, ci]
                im = ax.imshow(vel, cmap="viridis", aspect="auto",
                               vmin=vel_vmin, vmax=vel_vmax)
                # top row: column header (step count)
                title = f"{ns} steps" if si == 0 else ""
                if title:
                    ax.set_title(title, fontsize=9, fontweight="bold")
                # left column: row label (seed index)
                if ci == 0:
                    ax.set_ylabel(f"seed {si}", fontsize=8)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        plt.suptitle(
            f"DDIM step-count comparison  |  t_start={t_start}  eta={eta}",
            fontsize=11,
        )
        plt.tight_layout()
        cmp_path = out_dir / "comparison_steps.png"
        plt.savefig(cmp_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  comparison saved → {cmp_path}")

    # -----------------------------------------------------------------------
    # Comparison (original scheduler): same seeds × different step counts,
    # using standard DDIMScheduler.set_timesteps(N).
    # Each N gives a different effective t_start: t_start ≈ (N-1)/N * 1000.
    # -----------------------------------------------------------------------
    cmp_orig_cfg = cfg.get("comparison_original") or {}
    if cmp_orig_cfg.get("enabled", False):
        step_counts_orig   = [int(s) for s in cmp_orig_cfg.get("step_counts", [3, 6, 10, 20, 100])]
        n_cmp_orig_samples = int(cmp_orig_cfg.get("n_samples", 3))

        # Pre-compute the effective t_start for each N (for column labels)
        def _orig_tstart(n: int) -> int:
            ddim_sch.set_timesteps(n)
            return int(ddim_sch.timesteps[0].item())

        orig_tstarts = [_orig_tstart(n) for n in step_counts_orig]

        print(f"\n[comparison_original]  step_counts={step_counts_orig}")
        for ns, ts_ in zip(step_counts_orig, orig_tstarts):
            print(f"  N={ns:4d}  →  t_start={ts_}")

        def _orig_sample(z: torch.Tensor, num_steps: int) -> np.ndarray:
            """Generate with standard DDIMScheduler.set_timesteps(num_steps)."""
            ddim_sch.set_timesteps(num_steps)
            x = z.detach()
            with torch.no_grad():
                for t in ddim_sch.timesteps:
                    model_output = wrapper(x, t).sample
                    x = ddim_sch.step(
                        model_output, t, x,
                        eta=eta, use_clipped_model_output=False,
                    ).prev_sample
            return v_denorm_np(x)

        orig_grid: list[list[np.ndarray]] = []
        for si in range(n_cmp_orig_samples):
            z = torch.randn(
                1, 1, 70, 70,
                device=device, dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(seed + si),
            )
            row = []
            for ns in step_counts_orig:
                vel = _orig_sample(z, ns)
                row.append(vel)
                print(f"  seed {si}  steps={ns:4d}  "
                      f"range=[{vel.min():.0f}, {vel.max():.0f}] m/s")
            orig_grid.append(row)

        ncols_orig = len(step_counts_orig)
        nrows_orig = n_cmp_orig_samples
        fig3, axes3 = plt.subplots(
            nrows_orig, ncols_orig,
            figsize=(3.0 * ncols_orig, 2.9 * nrows_orig),
        )
        axes3 = np.array(axes3).reshape(nrows_orig, ncols_orig)

        for si in range(nrows_orig):
            for ci, (ns, ts_, vel) in enumerate(zip(step_counts_orig, orig_tstarts, orig_grid[si])):
                ax = axes3[si, ci]
                im = ax.imshow(vel, cmap="viridis", aspect="auto",
                               vmin=vel_vmin, vmax=vel_vmax)
                if si == 0:
                    ax.set_title(f"{ns} steps\n(t_start={ts_})", fontsize=8, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(f"seed {si}", fontsize=8)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        plt.suptitle(
            f"DDIM step-count comparison  |  standard scheduler  |  "
            f"z ~ N(0,I)  eta={eta}\n"
            f"(t_start shifts with N: larger N → t_start closer to 999)",
            fontsize=10,
        )
        plt.tight_layout()
        orig_path = out_dir / "comparison_steps_original.png"
        plt.savefig(orig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  original-scheduler comparison saved → {orig_path}")


if __name__ == "__main__":
    main()
