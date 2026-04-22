#!/usr/bin/env python3
"""
Curvevel-B velocity fields: DDIM sampling.

Default (training-aligned): ``OpenFWIUNetWrapper`` is the full 70×70 model; training saves ``model.pt`` (``torch.save``).
Uses ``DDIMPipelineOpenFWI`` (noise shape from ``wrapper.spatial``). Same forward as training; no post-crop.

``--raw-unet``: load bare ``UNet2DModel`` (72×72), optional center crop to 70×70.

Denormalization: ``v (m/s) = x * velocity_scale_m_s + velocity_center_m_s``.

Config auto-discovery order (when ``--config`` is omitted):
  1. ``<run_dir>/config_used.yaml``  (sibling of the checkpoint folder)
  2. ``training/configs/curvevel_b_ddpm.yaml``
  3. Hardcoded defaults (center=3000, scale=1500).

Example::

  cd Manifold_constrained_FWI
  python scripts/sample_ddim_curvevel_b.py \\
    --checkpoint training/runs/curvevel_b_ddpm_20260410_194949/checkpoint_epoch_0200
"""

from __future__ import annotations

import os
import sys
import warnings

# ── Suppress noisy import-time messages (TF / xformers / CUDA) ────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = _SCRIPTS_DIR.parent
_TRAINING_DIR = _MANIFOLD_ROOT / "training"
_REPO_ROOT = _MANIFOLD_ROOT.parent
_CURVE_VEL_B = _REPO_ROOT / "CurveVelB"
if str(_CURVE_VEL_B) not in sys.path:
    sys.path.insert(0, str(_CURVE_VEL_B))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

# ── Redirect OS-level stderr (fd 2) to /dev/null during library imports ────────
# This silences C++-level messages from TF/absl/CUDA that can't be caught by
# Python's warnings module. Python-level stderr is restored in the finally block
# so any real ImportError traceback is still visible.
_null_fd = os.open(os.devnull, os.O_WRONLY)
_saved_fd2 = os.dup(2)
_saved_py_stderr = sys.stderr
os.dup2(_null_fd, 2)
os.close(_null_fd)
sys.stderr = open(os.devnull, "w")
try:
    from diffusers_torch_compat import ensure_diffusers_custom_ops_safe  # noqa: E402
    ensure_diffusers_custom_ops_safe()
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, DDIMPipeline, DDPMScheduler, DDPMPipeline, UNet2DModel
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
    from diffusers.utils import is_torch_xla_available
    from diffusers.utils.torch_utils import randn_tensor
    from openfwi_unet_wrapper import OpenFWIUNetWrapper, load_openfwi_checkpoint
finally:
    sys.stderr.close()
    sys.stderr = _saved_py_stderr
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd2)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    _XLA_AVAILABLE = True
else:
    _XLA_AVAILABLE = False


class DDIMPipelineOpenFWI(DDIMPipeline):
    """DDIMPipeline with initial noise shaped by data grid ``wrapper.spatial`` (not inner UNet 72)."""

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: bool | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
    ) -> ImagePipelineOutput | tuple:
        w = self.unet
        uc = w.unet.config
        spatial = int(w.spatial)
        image_shape = (batch_size, uc.in_channels, spatial, spatial)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(
            image_shape,
            generator=generator,
            device=self.device,
            dtype=w.unet.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t).sample
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

            if _XLA_AVAILABLE:
                xm.mark_step()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def _load_script_config() -> dict:
    """Read scripts/config_sample_ddim.yaml (if present) → argparse-compatible defaults.

    Keys must match argparse dest names (underscores, not dashes).
    null / None values are ignored so the argparse built-in default applies.
    CLI arguments always override the config file.
    """
    cfg_path = _SCRIPTS_DIR / "config_sample_ddim.yaml"
    if not cfg_path.is_file():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {k.replace("-", "_"): v for k, v in raw.items() if v is not None}


def _default_checkpoint() -> Path:
    return (
        _TRAINING_DIR
        / "runs"
        / "curvevel_b_ddpm_20260410_194949"
        / "checkpoint_epoch_0200"
    )


def _default_out_dir_timestamped() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _MANIFOLD_ROOT / "exp" / "test_generation" / stamp


def _torch_dtype_from_unet_yaml_block(ucfg: dict) -> torch.dtype:
    s = str(ucfg.get("torch_dtype", "float32")).lower()
    return torch.float16 if s in ("float16", "fp16") else torch.float32


def load_ddim_pipeline_openfwi(
    checkpoint: Path, training_yaml: Path | None
) -> DDIMPipelineOpenFWI:
    """Load ``OpenFWIUNetWrapper`` + DDIM; 70×70 interface as in ``train_ddpm_curvevel_b.py``."""
    checkpoint = checkpoint.resolve()
    index = checkpoint / "model_index.json"
    if index.is_file():
        raise SystemExit("Full pipeline with model_index.json: use diffusers from_pretrained or custom loading.")

    model_pt = checkpoint / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"Missing {model_pt}; expected training checkpoint (torch.save of OpenFWIUNetWrapper).")
    td = torch.float32
    if training_yaml is not None and training_yaml.is_file():
        with open(training_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        td = _torch_dtype_from_unet_yaml_block(cfg["unet"])
    wrapper = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=td)
    ddpm_sched = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    ddim = DDIMScheduler.from_config(ddpm_sched.config)
    return DDIMPipelineOpenFWI(unet=wrapper, scheduler=ddim)


def load_ddim_pipeline_raw_unet(checkpoint: Path) -> DDIMPipeline:
    """Bare ``UNet2DModel`` (72×72); for ``--raw-unet`` only."""
    checkpoint = checkpoint.resolve()
    index = checkpoint / "model_index.json"
    if index.is_file():
        base = DDPMPipeline.from_pretrained(str(checkpoint))
        ddim = DDIMScheduler.from_config(base.scheduler.config)
        return DDIMPipeline(unet=base.unet, scheduler=ddim)

    unet = UNet2DModel.from_pretrained(str(checkpoint), subfolder="unet")
    ddpm_sched = DDPMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    ddim = DDIMScheduler.from_config(ddpm_sched.config)
    return DDIMPipeline(unet=unet, scheduler=ddim)


def np_images_to_velocity_m_s(
    np_01: np.ndarray,
    velocity_center_m_s: float,
    velocity_scale_m_s: float,
) -> np.ndarray:
    """output_type=np is [0,1] (H,W,C); map to [-1,1] then to m/s."""
    x = (np_01.astype(np.float64) - 0.5) * 2.0
    return x * velocity_scale_m_s + velocity_center_m_s


def crop_unet_inner_to_data_spatial(
    images: np.ndarray,
    *,
    inner: int,
    spatial: int,
) -> np.ndarray:
    """Match ``OpenFWIUNetWrapper.forward``: UNet inner×inner, take center spatial×spatial.

    When ``inner == spatial + 2`` (replicate pad), slice ``[1:spatial+1, 1:spatial+1]``.
    ``images``: (B, H, W, C) numpy.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected (B,H,W,C), got shape={images.shape}")
    h, w = int(images.shape[1]), int(images.shape[2])
    if h != inner or w != inner:
        return images
    if inner == spatial:
        return images
    if inner != spatial + 2:
        raise ValueError(
            f"OpenFWI crop only when inner=spatial+2; got inner={inner}, spatial={spatial}"
        )
    return images[:, 1 : spatial + 1, 1 : spatial + 1, :].copy()


def _velocity_norm_from_yaml(cfg_path: Path) -> tuple[float, float]:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", cfg)
    c = float(d.get("velocity_center_m_s", 3000.0))
    s = float(d.get("velocity_scale_m_s", 1500.0))
    return c, s


def _resolve_training_yaml_path(config_arg: str | None, run_dir: Path | None = None) -> Path:
    if config_arg is not None:
        cfg_yaml = Path(config_arg)
        if not cfg_yaml.is_absolute():
            c1 = (Path.cwd() / cfg_yaml).resolve()
            c2 = (_TRAINING_DIR / cfg_yaml).resolve()
            cfg_yaml = c1 if c1.is_file() else c2
        return cfg_yaml
    # Auto-discovery: 1) run_dir/config_used.yaml  2) training/configs/curvevel_b_ddpm.yaml
    if run_dir is not None:
        candidate = (run_dir / "config_used.yaml").resolve()
        if candidate.is_file():
            return candidate
    return (_TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curvevel-B: DDIM sampling from a training checkpoint (DDIMPipeline + DDIMScheduler).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="DDIM sampling; checkpoint scheduler is DDPM table used to build DDIMScheduler.",
    )
    script_cfg = _load_script_config()  # read early; applied after add_argument() calls
    opt = parser.add_argument_group(
        "Options",
        "Paths, DDIM hyperparameters, output, velocity denormalization; see defaults below.",
    )
    opt.add_argument(
        "--checkpoint",
        type=str,
        default=str(_default_checkpoint()),
        metavar="DIR",
        help="Checkpoint directory containing model.pt + scheduler/ "
             "(relative to cwd or training/; default: curvevel_b_ddpm_20260410_194949/checkpoint_epoch_0200).",
    )
    opt.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="YAML",
        help="Training YAML: data denorm and unet.config (build wrapper). "
        "Default training/configs/curvevel_b_ddpm.yaml if omitted; else checkpoint/unet/config.json.",
    )
    opt.add_argument(
        "--raw-unet",
        action="store_true",
        help="Skip OpenFWIUNetWrapper; load bare 72×72 UNet (use with --spatial / --no-crop).",
    )
    opt.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="Batch size for DDIM.",
    )
    opt.add_argument(
        "--steps",
        type=int,
        default=1000,
        metavar="T",
        help="DDIM denoising steps (num_inference_steps).",
    )
    opt.add_argument(
        "--eta",
        type=float,
        default=0.0,
        metavar="η",
        help="DDIM eta: 0 = deterministic; >0 adds stochasticity.",
    )
    opt.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="S",
        help="Random seed (torch / Generator).",
    )
    opt.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory; default <Manifold root>/exp/test_generation/<timestamp>/.",
    )
    opt.add_argument(
        "--velocity-center-m-s",
        type=float,
        default=None,
        metavar="V0",
        help="Denorm center (m/s); pair with --velocity-scale-m-s or read from --config.",
    )
    opt.add_argument(
        "--velocity-scale-m-s",
        type=float,
        default=None,
        metavar="VS",
        help="Denorm scale (m/s); pair with --velocity-center-m-s or read from --config.",
    )
    opt.add_argument(
        "--spatial",
        type=int,
        default=70,
        metavar="S0",
        help="With --raw-unet only: center-crop to S0×S0 (default 70).",
    )
    opt.add_argument(
        "--no-crop",
        action="store_true",
        help="With --raw-unet only: no crop, keep 72×72.",
    )
    # set_defaults MUST come after all add_argument() calls so it can update
    # each action's .default attribute (not just the internal _defaults dict).
    if script_cfg:
        parser.set_defaults(**script_cfg)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        cand_cwd = (Path.cwd() / ckpt).resolve()
        cand_train = (_TRAINING_DIR / ckpt).resolve()
        ckpt = cand_cwd if cand_cwd.is_dir() else cand_train
    else:
        ckpt = ckpt.resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    cfg_yaml = _resolve_training_yaml_path(args.config, run_dir=ckpt.parent)

    if args.velocity_center_m_s is not None and args.velocity_scale_m_s is not None:
        velocity_center_m_s = float(args.velocity_center_m_s)
        velocity_scale_m_s = float(args.velocity_scale_m_s)
    else:
        if args.velocity_center_m_s is not None or args.velocity_scale_m_s is not None:
            raise SystemExit("Provide both --velocity-center-m-s and --velocity-scale-m-s, or neither (use config).")
        if cfg_yaml.is_file():
            velocity_center_m_s, velocity_scale_m_s = _velocity_norm_from_yaml(cfg_yaml)
        else:
            velocity_center_m_s, velocity_scale_m_s = 3000.0, 1500.0

    if args.out_dir is None:
        out_dir = _default_out_dir_timestamped()
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir).resolve()
        else:
            out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[DDIM] num_inference_steps={args.steps} eta={args.eta} batch_size={args.batch_size} "
        f"seed={args.seed} raw_unet={args.raw_unet}\n"
        f"       checkpoint={ckpt}\n"
        f"       config={'(not found, using defaults)' if not cfg_yaml.is_file() else str(cfg_yaml)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    yaml_for_unet = cfg_yaml if cfg_yaml.is_file() else None

    if args.raw_unet:
        pipe = load_ddim_pipeline_raw_unet(ckpt)
        pipe = pipe.to(device)
        inner_grid = int(pipe.unet.config.sample_size)

        result = pipe(
            batch_size=args.batch_size,
            num_inference_steps=args.steps,
            eta=args.eta,
            generator=generator,
            output_type="np",
        )
        images = result.images

        if not args.no_crop:
            try:
                images = crop_unet_inner_to_data_spatial(
                    images, inner=inner_grid, spatial=int(args.spatial)
                )
            except ValueError as e:
                raise SystemExit(str(e)) from e

        meta_unet = {
            "network": "UNet2DModel",
            "raw_unet": True,
            "unet_config_sample_size": inner_grid,
            "crop_to_spatial": int(args.spatial) if not args.no_crop else None,
            "no_crop": bool(args.no_crop),
        }
    else:
        pipe = load_ddim_pipeline_openfwi(ckpt, yaml_for_unet)
        pipe = pipe.to(device)
        inner_grid = int(pipe.unet.unet.config.sample_size)
        data_spatial = int(pipe.unet.spatial)

        result = pipe(
            batch_size=args.batch_size,
            num_inference_steps=args.steps,
            eta=args.eta,
            generator=generator,
            output_type="np",
        )
        images = result.images

        meta_unet = {
            "network": "OpenFWIUNetWrapper",
            "raw_unet": False,
            "data_spatial": data_spatial,
            "inner_unet_sample_size": inner_grid,
        }

    meta = {
        "sampler": "DDIM",
        "checkpoint": str(ckpt),
        "training_yaml_used": str(cfg_yaml) if cfg_yaml.is_file() else None,
        **meta_unet,
        "output_hw": [int(images.shape[1]), int(images.shape[2])],
        "velocity_center_m_s": velocity_center_m_s,
        "velocity_scale_m_s": velocity_scale_m_s,
        "batch_size": args.batch_size,
        "ddim_num_inference_steps": args.steps,
        "ddim_eta": args.eta,
        "seed": args.seed,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "sample_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.save(out_dir / "samples_01.npy", images)

    vel = np_images_to_velocity_m_s(images, velocity_center_m_s, velocity_scale_m_s)
    np.save(out_dir / "velocity_m_s.npy", vel)

    n = min(args.batch_size, 4)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        im = ax.imshow(vel[i, :, :, 0], cmap="viridis", aspect="auto")
        ax.set_title(f"sample {i} (m/s)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(
        f"Curvevel-B DDIM (Manifold) η={args.eta}, steps={args.steps} — velocity (m/s)"
    )
    plt.tight_layout()
    fig_path = out_dir / "curvevel_b_generated.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
    print(f"Output dir: {out_dir}")
    print(f"Shapes: images={images.shape}, velocity_m_s={vel.shape}")


if __name__ == "__main__":
    main()
