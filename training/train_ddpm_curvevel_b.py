#!/usr/bin/env python3
"""
Curvevel-B DDPM training: aligned with diffusers ``examples/unconditional_image_generation/train_unconditional.py``
(Accelerate, ``get_scheduler``, ``accumulate``, grad clip, optional EMA / xformers / TensorBoard).
Dataset + ``OpenFWIUNetWrapper`` (``training/openfwi_unet_wrapper.py``) as the trainable 70×70 model;
epoch checkpoints save ``model.pt`` via ``torch.save``. Hyperparameters from YAML.

Velocity (m/s) default linear norm ``y = (x - 3000) / 1500`` to [-1, 1] (override in YAML).

Usage:
  accelerate launch train_ddpm_curvevel_b.py --config configs/curvevel_b_ddpm.yaml
  python train_ddpm_curvevel_b.py --config configs/curvevel_b_ddpm.yaml

GPU: YAML ``training.cuda_device`` (default 3) sets ``CUDA_VISIBLE_DEVICES`` before ``import torch`` if unset;
``--cuda-device N`` overrides YAML.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Call before first diffusers import: PyTorch 2.11+ vs diffusers flash-attn-3 custom op registration.
_TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_DIR.parent.parent
_CURVE_VEL_B = _REPO_ROOT / "CurveVelB"
if str(_CURVE_VEL_B) not in sys.path:
    sys.path.insert(0, str(_CURVE_VEL_B))
from diffusers_torch_compat import ensure_diffusers_custom_ops_safe  # noqa: E402

ensure_diffusers_custom_ops_safe()

if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))


def _apply_cuda_device_env_before_torch() -> None:
    """Set CUDA_VISIBLE_DEVICES before ``import torch`` from YAML or ``--cuda-device``."""
    import os

    import yaml

    training_dir = Path(__file__).resolve().parent
    default_cfg = training_dir / "configs" / "curvevel_b_ddpm.yaml"
    cfg_path = default_cfg
    cuda_cli: int | None = None
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            p = Path(sys.argv[i + 1])
            cfg_path = p.resolve()
            i += 2
            continue
        if sys.argv[i] in ("--cuda-device", "--cuda_device") and i + 1 < len(sys.argv):
            raw = sys.argv[i + 1].strip().lower()
            cuda_cli = int(raw.split(":")[-1]) if "cuda:" in raw else int(raw)
            i += 2
            continue
        i += 1

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        if cuda_cli is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_cli)
            return
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dev = cfg.get("training", {}).get("cuda_device")
        if dev is None:
            return
        s = str(dev).strip().lower()
        idx = int(s.split(":")[-1]) if "cuda:" in s else int(dev)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    except Exception:
        pass


_apply_cuda_device_env_before_torch()

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset

_MANIFOLD_ROOT = _TRAINING_DIR.parent

from openfwi_unet_wrapper import (
    OpenFWIUNetWrapper,
    load_openfwi_checkpoint,
    resolve_pretrained_model_pt,
    save_openfwi_checkpoint,
)

_DEFAULT_DATA = _MANIFOLD_ROOT / "data" / "Curvevel-B"
_DEFAULT_UNET_CFG = _CURVE_VEL_B / "CVB-2000" / "unet"
_DEFAULT_SCHEDULER_CFG = _CURVE_VEL_B / "CVB-2000" / "scheduler"


def _velocity_norm_from_data_cfg(dcfg: dict[str, Any]) -> tuple[float, float]:
    if "velocity_center_m_s" in dcfg or "velocity_scale_m_s" in dcfg:
        return float(dcfg.get("velocity_center_m_s", 3000.0)), float(dcfg.get("velocity_scale_m_s", 1500.0))
    if "label_min_m_s" in dcfg and "label_max_m_s" in dcfg:
        lo = float(dcfg["label_min_m_s"])
        hi = float(dcfg["label_max_m_s"])
        return (lo + hi) / 2.0, (hi - lo) / 2.0
    return 3000.0, 1500.0


def _build_openfwi_unet(
    ucfg: dict[str, Any],
    torch_dtype: torch.dtype,
    pretrained_resolved: Optional[Path],
    load_pretrained: bool,
) -> OpenFWIUNetWrapper:
    if load_pretrained and pretrained_resolved is not None:
        pt = resolve_pretrained_model_pt(pretrained_resolved)
        if pt is not None:
            return load_openfwi_checkpoint(pt, map_location="cpu", torch_dtype=torch_dtype)
        raise ValueError(
            "load_pretrained_weights=True requires pretrained_weights_path to be model.pt (or .pth) "
            "or a directory containing model.pt"
        )

    unet_config_yaml = ucfg.get("config")
    if unet_config_yaml is not None:
        unet = UNet2DModel.from_config(dict(unet_config_yaml))
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
        return OpenFWIUNetWrapper(unet)
    unet_cfg_dir = _resolve_path(ucfg.get("config_dir"), _DEFAULT_UNET_CFG)
    if load_pretrained:
        unet = UNet2DModel.from_pretrained(str(unet_cfg_dir), torch_dtype=torch_dtype)
    else:
        unet = UNet2DModel.from_config(UNet2DModel.load_config(str(unet_cfg_dir)))
        if torch_dtype is not None:
            unet = unet.to(dtype=torch_dtype)
    return OpenFWIUNetWrapper(unet)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Same as upstream train_unconditional for prediction_type == sample loss."""
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _resolve_path(p: Optional[str], default: Path) -> Path:
    if p is None or (isinstance(p, str) and str(p).strip().lower() in ("null", "none", "")):
        return default
    path = Path(p)
    if not path.is_absolute():
        path = (_TRAINING_DIR / path).resolve()
    return path


def _resolve_path_optional(p: Optional[str]) -> Optional[Path]:
    """Paths relative to ``training/``; null/empty returns None."""
    if p is None or (isinstance(p, str) and str(p).strip().lower() in ("null", "none", "")):
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (_TRAINING_DIR / path).resolve()
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CurveVelBNpyDataset(Dataset):
    """Flat index over ``model{i}.npy``; each file (N,1,70,70), default N=500.

    Linear norm ``y = (x_m_s - velocity_center_m_s) / velocity_scale_m_s``;
    default center=3000, scale=1500 maps [1500, 4500] to [-1, 1].
    """

    def __init__(
        self,
        data_root: Path,
        model_ids: list[int],
        samples_per_file: int,
        velocity_center_m_s: float = 3000.0,
        velocity_scale_m_s: float = 1500.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.model_ids = list(model_ids)
        self.velocity_center_m_s = float(velocity_center_m_s)
        self.velocity_scale_m_s = float(velocity_scale_m_s)
        if abs(self.velocity_scale_m_s) < 1e-12:
            raise ValueError("velocity_scale_m_s must be non-zero")
        self.samples_per_file = int(samples_per_file)

        self.paths: list[Path] = []
        for i in self.model_ids:
            p = data_root / f"model{i}.npy"
            if not p.is_file():
                raise FileNotFoundError(f"Missing data file: {p}")
            self.paths.append(p)

        self._mmaps: list[np.ndarray] = []
        for p in self.paths:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 4 or arr.shape[1:] != (1, 70, 70):
                raise ValueError(f"Expected shape (N,1,70,70), got {arr.shape} @ {p}")
            n = arr.shape[0]
            if n < self.samples_per_file:
                raise ValueError(f"{p} has only {n} samples; samples_per_file={self.samples_per_file}")
            self._mmaps.append(arr)

    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_file

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi = idx // self.samples_per_file
        si = idx % self.samples_per_file
        raw = np.array(self._mmaps[fi][si], dtype=np.float32)
        t = torch.from_numpy(raw).squeeze(0)
        y = (t - self.velocity_center_m_s) / self.velocity_scale_m_s
        y = y.clamp(-1.0, 1.0)
        return y.unsqueeze(0)


def _plot_loss_curve(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="train MSE")
    ax.plot(epochs, val_losses, label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def validate_epoch(
    accelerator: Accelerator,
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    loader: DataLoader,
    weight_dtype: torch.dtype,
    prediction_type: str,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    unwrapped = accelerator.unwrap_model(model)
    for batch in loader:
        clean = batch.to(accelerator.device, dtype=weight_dtype)
        b = clean.shape[0]
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (b,),
            device=accelerator.device,
            dtype=torch.long,
        )
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)
        with torch.autocast(device_type=accelerator.device.type, enabled=accelerator.mixed_precision != "no"):
            pred = unwrapped(noisy, timesteps).sample.float()

        if prediction_type == "epsilon":
            loss = F.mse_loss(pred, noise.float())
        elif prediction_type == "sample":
            alpha_t = _extract_into_tensor(
                noise_scheduler.alphas_cumprod, timesteps, (clean.shape[0], 1, 1, 1)
            )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = (snr_weights * F.mse_loss(pred, clean.float(), reduction="none")).mean()
        else:
            raise ValueError(prediction_type)
        total += loss.item() * b
        n += b
    return total / max(n, 1)


def _resolve_lr_scheduler_type(tcfg: dict[str, Any]) -> str:
    if tcfg.get("lr_scheduler_type"):
        return str(tcfg["lr_scheduler_type"])
    legacy = str(tcfg.get("lr_scheduler", "cosine"))
    if legacy == "cosine_epoch":
        return "cosine"
    if legacy == "none":
        return "constant"
    return legacy


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvevel-B DDPM training (Accelerate)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_TRAINING_DIR / "configs" / "curvevel_b_ddpm.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-range", type=int, nargs=2, default=None)
    parser.add_argument("--test-range", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Override YAML training.cuda_device (must match early bootstrap at process start).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_yaml(cfg_path)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.train_range is not None:
        cfg["data"]["train_model_id_range"] = [args.train_range[0], args.train_range[1]]
    if args.test_range is not None:
        cfg["data"]["test_model_id_range"] = [args.test_range[0], args.test_range[1]]
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)

    data_root = _resolve_path(cfg["data"].get("root"), _DEFAULT_DATA)
    tr = cfg["data"]["train_model_id_range"]
    te = cfg["data"]["test_model_id_range"]
    train_ids = list(range(tr[0], tr[1] + 1))
    test_ids = list(range(te[0], te[1] + 1))
    samples_per_file = int(cfg["data"]["samples_per_file"])
    dcfg = cfg["data"]
    velocity_center_m_s, velocity_scale_m_s = _velocity_norm_from_data_cfg(dcfg)
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"].get("pin_memory", True))

    ucfg = cfg["unet"]
    load_pretrained = bool(ucfg.get("load_pretrained_weights", False))
    torch_dtype_str = str(ucfg.get("torch_dtype", "float32")).lower()
    torch_dtype = torch.float16 if torch_dtype_str in ("float16", "fp16") else torch.float32
    unet_config_yaml = ucfg.get("config")
    pretrained_weights_resolved = _resolve_path_optional(ucfg.get("pretrained_weights_path"))

    sched_block = cfg["noise_scheduler"]
    sched_config_yaml = sched_block.get("config")
    sched_cfg_dir = _resolve_path(sched_block.get("config_dir"), _DEFAULT_SCHEDULER_CFG)

    tcfg = cfg["training"]
    if "mixed_precision" not in tcfg:
        tcfg["mixed_precision"] = "fp16" if tcfg.get("amp", True) else "no"
    out_root = _resolve_path(tcfg.get("output_root"), _TRAINING_DIR / "runs")
    run_name = str(tcfg.get("run_name", "curvevel_b_ddpm"))
    seed = int(tcfg["seed"])
    batch_size = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["lr"])
    weight_decay = float(tcfg["weight_decay"])
    betas = tuple(float(x) for x in tcfg["betas"])
    adam_epsilon = float(tcfg.get("adam_epsilon", 1e-8))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 1))
    mixed_precision = str(tcfg.get("mixed_precision", "no"))
    lr_scheduler_type = _resolve_lr_scheduler_type(tcfg)
    lr_warmup_steps = int(tcfg.get("lr_warmup_steps", 500))
    prediction_type = str(tcfg.get("prediction_type", "epsilon"))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = int(tcfg.get("log_every", 50))
    val_every = int(tcfg.get("val_every", 1))
    save_every = int(tcfg.get("save_every", 5))
    save_best = bool(tcfg.get("save_best", True))
    log_with = str(tcfg.get("log_with", "tensorboard")).lower()
    logging_subdir = str(tcfg.get("logging_subdir", "logs"))
    use_ema = bool(tcfg.get("use_ema", False))
    ema_inv_gamma = float(tcfg.get("ema_inv_gamma", 1.0))
    ema_power = float(tcfg.get("ema_power", 0.75))
    ema_max_decay = float(tcfg.get("ema_max_decay", 0.9999))
    enable_xformers = bool(tcfg.get("enable_xformers", False))
    accelerator_checkpoint_steps = tcfg.get("accelerator_checkpoint_steps")
    checkpoints_total_limit = tcfg.get("checkpoints_total_limit")
    resume_from_checkpoint = tcfg.get("resume_from_checkpoint")

    set_seed(seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(cfg_path, run_dir / "config_used.yaml")

    logging_dir = str(run_dir / logging_subdir)
    project_config = ProjectConfiguration(project_dir=str(run_dir), logging_dir=logging_dir)

    log_with_arg: Optional[str] = None
    if log_with == "tensorboard":
        if is_tensorboard_available():
            log_with_arg = "tensorboard"
        else:
            print("tensorboard not installed; disabling experiment_tracker")
    elif log_with == "wandb":
        if is_wandb_available():
            log_with_arg = "wandb"
        else:
            print("wandb not installed; disabling experiment_tracker")

    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200))]
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision if mixed_precision != "no" else None,
        log_with=log_with_arg,
        project_config=project_config,
        kwargs_handlers=kwargs_handlers,
    )

    if accelerator.is_main_process:
        print(f"Output dir: {run_dir}")

    train_ds = CurveVelBNpyDataset(
        data_root,
        train_ids,
        samples_per_file,
        velocity_center_m_s=velocity_center_m_s,
        velocity_scale_m_s=velocity_scale_m_s,
    )
    test_ds = CurveVelBNpyDataset(
        data_root,
        test_ids,
        samples_per_file,
        velocity_center_m_s=velocity_center_m_s,
        velocity_scale_m_s=velocity_scale_m_s,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_openfwi_unet(ucfg, torch_dtype, pretrained_weights_resolved, load_pretrained)

    if sched_config_yaml is not None:
        noise_scheduler = DDPMScheduler.from_config(sched_config_yaml)
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(str(sched_cfg_dir))

    if enable_xformers and is_xformers_available():
        import xformers

        xv = version.parse(xformers.__version__)
        if xv == version.parse("0.0.16"):
            print(
                "xFormers 0.0.16 may be unstable on some GPUs; consider upgrading to >= 0.0.17."
            )
        model.unet.enable_xformers_memory_efficient_attention()
    elif enable_xformers:
        raise RuntimeError("enable_xformers is true but xformers is not installed")

    ema_model: Optional[EMAModel] = None
    if use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=ema_inv_gamma,
            power=ema_power,
            model_cls=UNet2DModel,
            model_config=model.unet.config,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=adam_epsilon,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * grad_accum,
        num_training_steps=num_training_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if ema_model is not None:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process and log_with_arg is not None:
        accelerator.init_trackers(run_name, config={"data_root": str(data_root)})

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if resume_from_checkpoint and str(resume_from_checkpoint).strip():
        load_path = Path(str(resume_from_checkpoint)).expanduser().resolve()
        if load_path.is_dir():
            try:
                accelerator.load_state(str(load_path))
                accelerator.print(f"Resumed Accelerate checkpoint: {load_path}")
                if load_path.name.startswith("checkpoint-"):
                    try:
                        global_step = int(load_path.name.split("-")[1])
                        resume_global_step = global_step * grad_accum
                        first_epoch = resume_global_step // max(num_update_steps_per_epoch, 1)
                        resume_step = resume_global_step % (
                            max(num_update_steps_per_epoch, 1) * max(grad_accum, 1)
                        )
                    except (ValueError, IndexError):
                        pass
            except Exception as e:
                accelerator.print(f"Checkpoint resume failed; training from scratch: {e}")

    history: list[dict[str, Any]] = []
    best_val = float("inf")

    if accelerator.is_main_process:
        print(
            f"Train samples: {len(train_ds)}, val: {len(test_ds)}, "
            f"batch_size={batch_size}, grad_accum={grad_accum}, "
            f"optimizer steps per epoch ~ {num_update_steps_per_epoch}"
        )

    for epoch in range(first_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for step, batch in enumerate(train_loader):
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            clean = batch.to(dtype=weight_dtype)
            noise = torch.randn(clean.shape, dtype=weight_dtype, device=clean.device)
            bsz = clean.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=clean.device,
            ).long()

            noisy = noise_scheduler.add_noise(clean, noise, timesteps)

            with accelerator.accumulate(model):
                model_output = model(noisy, timesteps).sample

                if prediction_type == "epsilon":
                    loss = F.mse_loss(model_output.float(), noise.float())
                elif prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = (
                        snr_weights
                        * F.mse_loss(model_output.float(), clean.float(), reduction="none")
                    ).mean()
                else:
                    raise ValueError(f"Unsupported prediction_type: {prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().item() * bsz
            total_n += bsz

            if accelerator.sync_gradients:
                if ema_model is not None:
                    ema_model.step(accelerator.unwrap_model(model).parameters())
                global_step += 1

                if accelerator.is_main_process and log_every > 0 and global_step % log_every == 0:
                    print(
                        f"  global_step={global_step} loss={loss.item():.6f} "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                accelerator.log(
                    {
                        "train/loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

                if (
                    accelerator.is_main_process
                    and accelerator_checkpoint_steps is not None
                    and int(accelerator_checkpoint_steps) > 0
                    and global_step % int(accelerator_checkpoint_steps) == 0
                ):
                    save_path = run_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    if checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                            key=lambda x: int(x.name.split("-")[1]),
                        )
                        if len(ckpts) > int(checkpoints_total_limit):
                            for rm in ckpts[: len(ckpts) - int(checkpoints_total_limit)]:
                                shutil.rmtree(rm, ignore_errors=True)

        train_loss_epoch = total_loss / max(total_n, 1)

        val_loss = float("nan")
        if val_every > 0 and (epoch + 1) % val_every == 0 and accelerator.is_main_process:
            val_loss = validate_epoch(
                accelerator, model, noise_scheduler, test_loader, weight_dtype, prediction_type
            )

        accelerator.wait_for_everyone()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss,
            "lr": lr_scheduler.get_last_lr()[0],
            "global_step_end": global_step,
        }
        history.append(row)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1}/{epochs} train_loss={train_loss_epoch:.6f} "
                f"val_loss={val_loss} lr={lr_scheduler.get_last_lr()[0]:.2e}"
            )
            with open(run_dir / "loss_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            unwrapped = accelerator.unwrap_model(model)
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                ckpt_dir = run_dir / f"checkpoint_epoch_{epoch + 1:04d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_openfwi_checkpoint(unwrapped, ckpt_dir / "model.pt")
                noise_scheduler.save_pretrained(ckpt_dir / "scheduler")

            if save_best and val_every > 0 and not np.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_dir = run_dir / "checkpoint_best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                save_openfwi_checkpoint(unwrapped, best_dir / "model.pt")
                noise_scheduler.save_pretrained(best_dir / "scheduler")
                with open(best_dir / "best.json", "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch + 1, "val_loss": val_loss}, f, indent=2)

    accelerator.wait_for_everyone()
    accelerator.end_training()

    if accelerator.is_main_process:
        epochs_list = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        _plot_loss_curve(epochs_list, train_losses, val_losses, run_dir / "loss_curve.png")

        unet_meta: dict[str, Any] = {}
        if unet_config_yaml is not None:
            unet_meta["unet_config"] = "embedded_in_yaml"
            if pretrained_weights_resolved is not None:
                unet_meta["pretrained_weights_path"] = str(pretrained_weights_resolved)
        else:
            unet_meta["unet_config_dir"] = str(
                _resolve_path(ucfg.get("config_dir"), _DEFAULT_UNET_CFG)
            )

        sched_meta: dict[str, Any] = {}
        if sched_config_yaml is not None:
            sched_meta["noise_scheduler_config"] = "embedded_in_yaml"
        else:
            sched_meta["scheduler_config_dir"] = str(sched_cfg_dir)

        meta = {
            "data_root": str(data_root),
            "velocity_normalization": {
                "formula": "y = (x_m_s - center) / scale",
                "velocity_center_m_s": velocity_center_m_s,
                "velocity_scale_m_s": velocity_scale_m_s,
            },
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            **unet_meta,
            **sched_meta,
            "load_pretrained_weights": load_pretrained,
            "hyperparams": {
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": list(betas),
                "adam_epsilon": adam_epsilon,
                "lr_scheduler_type": lr_scheduler_type,
                "lr_warmup_steps": lr_warmup_steps,
                "gradient_accumulation_steps": grad_accum,
                "mixed_precision": mixed_precision,
                "prediction_type": prediction_type,
                "grad_clip": grad_clip,
                "use_ema": use_ema,
            },
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"Done. Best val loss ~ {best_val}; output in {run_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
