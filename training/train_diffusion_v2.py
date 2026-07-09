#!/usr/bin/env python3
"""扩散模型训练 v2：现有 UNet 架构 + RED-DiffEq 训练配方.

与 v1 训练的关键变更:
  - Sigmoid 噪声调度 (start=-3, end=3, tau=1)
  - Adam (无 weight decay), lr=2e-4, constant, betas=(0.9, 0.99)
  - EMA decay=0.995 (diffusers EMAModel)
  - 按 total_steps 训练 (非 epochs)
  - 支持所有 4 个 OpenFWI 数据集

用法:
  uv run accelerate launch Manifold_constrained_FWI/training/train_diffusion_v2.py \
      --config Manifold_constrained_FWI/training/configs/diffusion_v2.yaml

  在 YAML 中修改 ``data.dataset`` 切换数据集:
    CurveFault-B | CurveVel-B | FlatFault-B | FlatVel-B
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

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset

_TRAINING_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = _TRAINING_DIR.parent
_ORIGINAL_TRAINING = _TRAINING_DIR  # same dir, to import wrapper
sys.path.insert(0, str(_TRAINING_DIR))

from openfwi_unet_wrapper import OpenFWIUNetWrapper, save_openfwi_checkpoint


# =============================================================================
# Sigmoid 噪声调度 (与 RED-DiffEq 一致: start=-3, end=3, tau=1)
# =============================================================================
def build_sigmoid_betas(num_timesteps: int = 1000, start: float = -3.0,
                        end: float = 3.0, tau: float = 1.0) -> np.ndarray:
    """Sigmoid beta schedule matching denoising-diffusion-pytorch."""
    steps = num_timesteps + 1
    t = np.linspace(0, 1, steps, dtype=np.float64)
    v_start = 1.0 / (1.0 + np.exp(-start / tau))
    v_end   = 1.0 / (1.0 + np.exp(-end   / tau))
    alpha_cumprod = (-1.0 / (1.0 + np.exp(-(t * (end - start) + start) / tau))
                     + v_end) / (v_end - v_start)
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1.0 - alpha_cumprod[1:] / alpha_cumprod[:-1]
    return np.clip(betas, 0.0, 0.999).astype(np.float32)


# =============================================================================
# 统一数据集：支持 prefix 和 plain 两种文件命名
# =============================================================================
class VelocityDataset(Dataset):
    """统一速度场数据集，支持两种文件命名模式:

    - prefix 模式:  ``vel{prefix}_{suffix}_{i}.npy``
    - plain 模式:   ``model{i}.npy``

    速度归一化: y = (x_m_s - center) / scale, 后 clamp 到 [-1, 1].
    """

    def __init__(
        self,
        data_root: Path,
        file_prefixes: list[int] | None,
        file_suffix: str | None,
        index_ids: list[int],
        samples_per_file: int,
        velocity_center_m_s: float = 3000.0,
        velocity_scale_m_s: float = 1500.0,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.velocity_center_m_s = float(velocity_center_m_s)
        self.velocity_scale_m_s = float(velocity_scale_m_s)
        self.samples_per_file = int(samples_per_file)

        self.paths: list[Path] = []
        if file_prefixes is not None and len(file_prefixes) > 0:
            # prefix 模式: vel{prefix}_{suffix}_{i}.npy
            if file_suffix is None:
                raise ValueError("file_suffix required for prefix mode")
            for prefix in file_prefixes:
                for idx in index_ids:
                    p = self.data_root / f"vel{prefix}_{file_suffix}_{idx}.npy"
                    if not p.is_file():
                        raise FileNotFoundError(f"Missing: {p}")
                    self.paths.append(p)
        else:
            # plain 模式: model{i}.npy
            for idx in index_ids:
                p = self.data_root / f"model{idx}.npy"
                if not p.is_file():
                    raise FileNotFoundError(f"Missing: {p}")
                self.paths.append(p)

        # mmap 加速
        self._mmaps: list[np.ndarray] = []
        for p in self.paths:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 4 or arr.shape[1] not in (1,):
                raise ValueError(f"Bad shape {arr.shape} @ {p}")
            if arr.shape[0] < self.samples_per_file:
                raise ValueError(
                    f"{p} has {arr.shape[0]} samples, need {self.samples_per_file}"
                )
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


# =============================================================================
# 工具
# =============================================================================
def _resolve_path(p: Optional[str], default: Path) -> Path:
    if p is None or str(p).strip().lower() in ("null", "none", ""):
        return default
    path = Path(p)
    return path if path.is_absolute() else (_TRAINING_DIR / path).resolve()


def _plot_loss(train_steps: list[int], train_losses: list[float],
               val_steps: list[int], val_losses: list[float],
               out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_steps, train_losses, label="train loss", alpha=0.7, linewidth=0.5)
    ax.plot(val_steps, val_losses, "o-", label="val loss", markersize=3)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def validate(
    accelerator: Accelerator, model, noise_scheduler: DDPMScheduler,
    loader: DataLoader, weight_dtype: torch.dtype,
) -> float:
    model.eval()
    total, n = 0.0, 0
    unwrapped = accelerator.unwrap_model(model)
    for batch in loader:
        clean = batch.to(accelerator.device, dtype=weight_dtype)
        b = clean.shape[0]
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (b,),
            device=accelerator.device, dtype=torch.long,
        )
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)
        pred = unwrapped(noisy, timesteps).sample.float()
        loss = F.mse_loss(pred, noise.float())
        total += loss.item() * b
        n += b
    return total / max(n, 1)


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="DDPM v2 training (RED-DiffEq recipe)")
    parser.add_argument("--config", type=str,
                        default=str(_TRAINING_DIR / "configs" / "diffusion_v2.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_yaml(cfg_path)

    # ---- resolve config ----
    dcfg  = cfg["data"]
    ucfg  = cfg["unet"]
    scfg  = cfg["noise_schedule"]
    tcfg  = cfg["training"]

    dataset_name = str(dcfg["dataset"])
    data_root = _MANIFOLD_ROOT / "data" / dataset_name

    # multi-prefix vs plain
    file_prefixes: list[int] | None = None
    file_suffix: str | None = None
    raw_prefixes = dcfg.get("file_prefixes")
    if raw_prefixes is not None and len(raw_prefixes) > 0:
        file_prefixes = [int(p) for p in raw_prefixes]
        file_suffix = str(dcfg.get("file_suffix", "1")) if dcfg.get("file_suffix") is not None else None

    train_ids = list(range(dcfg["train_index_range"][0], dcfg["train_index_range"][1] + 1))
    test_ids  = list(range(dcfg["test_index_range"][0],  dcfg["test_index_range"][1] + 1))
    samples_per_file = int(dcfg["samples_per_file"])
    velocity_center_m_s = float(dcfg.get("velocity_center_m_s", 3000.0))
    velocity_scale_m_s  = float(dcfg.get("velocity_scale_m_s", 1500.0))
    num_workers = int(dcfg.get("num_workers", 4))
    pin_memory  = bool(dcfg.get("pin_memory", True))

    # training params
    seed           = int(tcfg["seed"])
    batch_size     = int(tcfg["batch_size"])
    total_steps    = int(tcfg["total_steps"])
    lr             = float(tcfg["lr"])
    weight_decay   = float(tcfg.get("weight_decay", 0.0))
    betas          = tuple(float(x) for x in tcfg["betas"])
    grad_accum     = int(tcfg.get("gradient_accumulation_steps", 1))
    mixed_precision = str(tcfg.get("mixed_precision", "fp16"))
    prediction_type = str(tcfg.get("prediction_type", "epsilon"))
    grad_clip      = float(tcfg.get("grad_clip", 1.0))
    log_every      = int(tcfg.get("log_every", 100))
    val_every      = int(tcfg.get("val_every", 1000))
    save_every     = int(tcfg.get("save_every", 50000))
    use_ema        = bool(tcfg.get("use_ema", True))
    ema_decay      = float(tcfg.get("ema_decay", 0.995))
    enable_xformers = bool(tcfg.get("enable_xformers", False))
    checkpoint_steps = tcfg.get("checkpoint_steps")
    checkpoints_total_limit = tcfg.get("checkpoints_total_limit")
    resume_from_checkpoint = tcfg.get("resume_from_checkpoint")

    run_name = tcfg.get("run_name") or f"{dataset_name}_v2"
    out_root = _resolve_path(tcfg.get("output_root"), _TRAINING_DIR / "runs_v2")

    # ---- setup ----
    set_seed(seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(cfg_path, run_dir / "config_used.yaml")

    project_config = ProjectConfiguration(
        project_dir=str(run_dir), logging_dir=str(run_dir / "logs")
    )
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200))]
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision if mixed_precision != "no" else None,
        project_config=project_config,
        kwargs_handlers=kwargs_handlers,
    )

    if accelerator.is_main_process:
        print(f"[v2] dataset={dataset_name}  run_dir={run_dir}")
        print(f"[v2] total_steps={total_steps}  batch={batch_size}  lr={lr}  "
              f"ema={use_ema}  noise=sigmoid")

    # ---- datasets ----
    train_ds = VelocityDataset(
        data_root, file_prefixes, file_suffix, train_ids, samples_per_file,
        velocity_center_m_s=velocity_center_m_s, velocity_scale_m_s=velocity_scale_m_s,
    )
    test_ds = VelocityDataset(
        data_root, file_prefixes, file_suffix, test_ids, samples_per_file,
        velocity_center_m_s=velocity_center_m_s, velocity_scale_m_s=velocity_scale_m_s,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if accelerator.is_main_process:
        print(f"[v2] train_samples={len(train_ds)}  val_samples={len(test_ds)}  "
              f"batches_per_epoch≈{len(train_loader)}")

    # ---- model (same architecture as v1) ----
    torch_dtype = torch.float32
    if ucfg.get("torch_dtype", "float32") in ("float16", "fp16"):
        torch_dtype = torch.float16

    unet = UNet2DModel.from_config(dict(ucfg["config"]))
    if torch_dtype is not None:
        unet = unet.to(dtype=torch_dtype)
    model = OpenFWIUNetWrapper(unet)

    if enable_xformers:
        try:
            model.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            print("[v2] xformers not available, skipping")

    # ---- sigmoid noise schedule ----
    betas_np = build_sigmoid_betas(
        num_timesteps=int(scfg["num_train_timesteps"]),
        start=float(scfg.get("sigmoid_start", -3)),
        end=float(scfg.get("sigmoid_end", 3)),
        tau=float(scfg.get("sigmoid_tau", 1)),
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(scfg["num_train_timesteps"]),
        trained_betas=betas_np.tolist(),
        prediction_type=prediction_type,
        clip_sample=False,
    )

    # ---- optimizer: Adam, no weight decay (RED-DiffEq recipe) ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,
    )

    # ---- EMA ----
    ema_model: Optional[EMAModel] = None
    if use_ema:
        ema_model = EMAModel(
            model.parameters(), decay=ema_decay,
            model_cls=UNet2DModel, model_config=model.unet.config,
        )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ---- training state ----
    global_step = 0
    start_step = 0

    if resume_from_checkpoint and str(resume_from_checkpoint).strip():
        load_path = Path(str(resume_from_checkpoint)).expanduser().resolve()
        if load_path.is_dir():
            try:
                accelerator.load_state(str(load_path))
                accelerator.print(f"[v2] Resumed from {load_path}")
                if load_path.name.startswith("step-"):
                    start_step = int(load_path.name.split("-")[1])
                    global_step = start_step
            except Exception as e:
                accelerator.print(f"[v2] Resume failed: {e}")

    # ---- training loop (step-based, not epoch-based) ----
    train_iter = iter(train_loader)
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    best_val = float("inf")

    if accelerator.is_main_process:
        print(f"[v2] Starting training from step {start_step} to {total_steps}")

    model.train()
    for step in range(start_step, total_steps):
        # get next batch (infinite iterator)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        clean = batch.to(dtype=weight_dtype)
        bsz = clean.shape[0]
        noise = torch.randn(clean.shape, dtype=weight_dtype, device=clean.device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,),
            device=clean.device, dtype=torch.long,
        )
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)

        with accelerator.accumulate(model):
            with torch.autocast(
                device_type=accelerator.device.type,
                enabled=accelerator.mixed_precision != "no",
            ):
                model_output = model(noisy, timesteps).sample
            loss = F.mse_loss(model_output.float(), noise.float())
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.detach().item()
        train_losses.append(loss_val)

        if accelerator.sync_gradients:
            if ema_model is not None:
                ema_model.step(accelerator.unwrap_model(model).parameters())
            global_step += 1

            if accelerator.is_main_process and log_every > 0 and global_step % log_every == 0:
                recent = np.mean(train_losses[-100:])
                print(f"[v2] step={global_step}/{total_steps}  "
                      f"loss(avg100)={recent:.6f}  loss(cur)={loss_val:.6f}")

        # ---- validation ----
        if (global_step > 0 and global_step % val_every == 0
                and accelerator.is_main_process):
            val_loss = validate(
                accelerator, model, noise_scheduler, test_loader, weight_dtype,
            )
            val_steps.append(global_step)
            val_losses.append(val_loss)
            print(f"[v2] step={global_step}  val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_dir = run_dir / "checkpoint_best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                if ema_model is not None:
                    ema_model.store(unwrapped.parameters())
                    ema_model.copy_to(unwrapped.parameters())
                save_openfwi_checkpoint(unwrapped, best_dir / "model.pt")
                noise_scheduler.save_pretrained(best_dir / "scheduler")
                if ema_model is not None:
                    ema_model.restore(unwrapped.parameters())
                with open(best_dir / "best.json", "w") as f:
                    json.dump({"step": global_step, "val_loss": val_loss}, f)

        # ---- save checkpoint ----
        if (global_step > 0 and save_every > 0
                and global_step % save_every == 0 and accelerator.is_main_process):
            ckpt_dir = run_dir / f"step-{global_step:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            if ema_model is not None:
                ema_model.store(unwrapped.parameters())
                ema_model.copy_to(unwrapped.parameters())
            save_openfwi_checkpoint(unwrapped, ckpt_dir / "model.pt")
            noise_scheduler.save_pretrained(ckpt_dir / "scheduler")
            if ema_model is not None:
                ema_model.restore(unwrapped.parameters())
            print(f"[v2] checkpoint saved: {ckpt_dir}")

        if accelerator.is_main_process and checkpoint_steps and global_step % int(checkpoint_steps) == 0:
            save_path = run_dir / f"state-step-{global_step:07d}"
            accelerator.save_state(str(save_path))

    # ---- final save ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # final EMA-smoothed checkpoint
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        if ema_model is not None:
            ema_model.store(unwrapped.parameters())
            ema_model.copy_to(unwrapped.parameters())
        save_openfwi_checkpoint(unwrapped, final_dir / "model.pt")
        noise_scheduler.save_pretrained(final_dir / "scheduler")
        if ema_model is not None:
            ema_model.restore(unwrapped.parameters())

        # loss plot
        _plot_loss(
            list(range(len(train_losses))), train_losses,
            val_steps, val_losses,
            run_dir / "loss.png",
        )

        # metadata
        meta = {
            "dataset": dataset_name,
            "train_samples": len(train_ds),
            "val_samples": len(test_ds),
            "total_steps": total_steps,
            "best_val_loss": best_val,
            "final_step": global_step,
            "hyperparams": {
                "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                "betas": list(betas), "prediction_type": prediction_type,
                "mixed_precision": mixed_precision, "grad_clip": grad_clip,
                "use_ema": use_ema, "ema_decay": ema_decay,
                "noise_schedule": "sigmoid",
                "sigmoid_start": scfg.get("sigmoid_start", -3),
                "sigmoid_end": scfg.get("sigmoid_end", 3),
                "sigmoid_tau": scfg.get("sigmoid_tau", 1),
            },
        }
        with open(run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"\n[v2] Done! best_val={best_val:.6f}  output={run_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
