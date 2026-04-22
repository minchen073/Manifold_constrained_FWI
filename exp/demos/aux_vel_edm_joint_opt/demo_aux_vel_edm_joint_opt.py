"""
辅助速度场 v_aux + 生成潜变量 z 的联合优化，以及传统 TV / Tikhonov 正则化 FWI 对照。

- **FWI（TV / Tikhonov）**：数据项 + 正则，初值为真值速度的高斯光滑。

- **联合（joint）**：同时优化 v_aux（同上初值）与 z，
  v_gen = edm_sampler_ode_latentgrad(EDM, z·σ_max)，
  L = L_wave(v_aux) + λ_coup·MSE(v_gen(z), v_aux)（均在 μ 空间）。
  无正则化项；当 λ_coup = 0 时不对 z 求梯度，仅 v_aux 更新（退化为普通数据驱动 FWI）。

运行（Manifold_constrained_FWI 目录）:
  uv run python exp/aux_vel_edm_joint_opt/demo_aux_vel_edm_joint_opt.py \\
    --config exp/aux_vel_edm_joint_opt/config_aux_vel_edm_joint_opt.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]  # exp/aux_vel_edm_joint_opt -> exp -> Manifold_constrained_FWI

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
from src.core.generate import edm_sampler_ode_latentgrad
from src.core.loss import WavefieldLoss
from src.seismic import seismic_master_forward_modeling

try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def v_denormalize(v_norm: torch.Tensor) -> torch.Tensor:
    return v_norm * 1500.0 + 3000.0


def velocity_mae_ssim(target: torch.Tensor, pred: torch.Tensor):
    t = target.detach().float()
    p = pred.detach().float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    if p.dim() == 2:
        p = p.unsqueeze(0).unsqueeze(0)
    elif p.dim() == 3:
        p = p.unsqueeze(0)
    if p.shape != t.shape:
        p = p.reshape(t.shape)
    mae_norm = torch.mean(torch.abs(t - p)).item()
    ssim_v = pytorch_ssim.ssim(t, p, window_size=11, size_average=True).item()
    mae_phys = torch.mean(torch.abs(v_denormalize(t) - v_denormalize(p))).item()
    return mae_norm, mae_phys, ssim_v


def fmt_metrics(mae_n: float, mae_p: float, ssim: float) -> str:
    return f"MAE_n={mae_n:.4f}, MAE={mae_p:.1f} m/s, SSIM={ssim:.4f}"


def load_dataset_sample(data_path: str, file_index: int, sample_index: int, device):
    data_file = Path(data_path) / f"model{file_index}.npy"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.load(data_file)  # (N, 1, 70, 70), physical velocity m/s
    target = data[sample_index]
    if target.ndim == 3:
        target = target[0]  # (70, 70)
    elif target.ndim == 4:
        target = target[0, 0]
    target_norm = (target.astype(np.float32) - 3000.0) / 1500.0
    t = torch.from_numpy(target_norm).float().to(device)
    return t


def load_observed_wavefield(data_path: str, file_index: int, sample_index: int, device):
    data_file = Path(data_path) / f"data{file_index}.npy"
    if not data_file.exists():
        raise FileNotFoundError(f"Wavefield data not found: {data_file}")
    data = np.load(data_file)  # (N, 5, 1000, 70)
    wf = torch.from_numpy(data[sample_index].astype(np.float32)).float().to(device)
    return wf


def forward_velocity_to_wavefield(v_norm: torch.Tensor, img_res: int) -> torch.Tensor:
    """归一化速度 μ → 模拟波场 (5, 1000, 70)。物理参数固定于 wave_equation_forward.py 全局变量。"""
    velocity_physical = v_denormalize(v_norm).clamp(1500.0, 4500.0)
    velocity_2d = velocity_physical.squeeze().reshape(img_res, img_res).to(v_norm.device)
    sim = seismic_master_forward_modeling(velocity_2d)
    return sim


def build_wavefield_loss(loss_type: str, w2_cfg: dict) -> WavefieldLoss:
    lt = str(loss_type).lower().strip()
    dt = 0.001  # fixed by wave_equation_forward.py
    if lt in ("wavefield_l1", "l1"):
        return WavefieldLoss(loss_type="l1", dt=dt)
    if lt in ("wavefield_mse", "mse"):
        return WavefieldLoss(loss_type="mse", dt=dt)
    if lt in ("wavefield_l2_sq", "l2_sq"):
        return WavefieldLoss(loss_type="l2_sq", dt=dt)
    if lt == "w2_per_trace":
        return WavefieldLoss(
            loss_type="w2_per_trace",
            dt=dt,
            normalize_type=w2_cfg.get("normalize_type", "softplus"),
            b=float(w2_cfg.get("b", 0.1)),
        )
    raise ValueError(f"Unknown loss_type {loss_type!r}")


def total_variation_loss(mu: torch.Tensor) -> torch.Tensor:
    diff_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])
    diff_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])
    tv_x = diff_x.view(diff_x.shape[0], -1).mean(dim=1)
    tv_y = diff_y.view(diff_y.shape[0], -1).mean(dim=1)
    return tv_x + tv_y


def tikhonov_loss(mu: torch.Tensor) -> torch.Tensor:
    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :]
    l2_x = (diff_x**2).view(diff_x.shape[0], -1).mean(dim=1)
    l2_y = (diff_y**2).view(diff_y.shape[0], -1).mean(dim=1)
    return l2_x + l2_y


FWI_LOSS_TYPES = ("wavefield_l1", "l1", "wavefield_mse", "mse", "wavefield_l2_sq", "l2_sq", "w2_per_trace")


def _vel_to_rgb_uint8(arr: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """归一化速度场 [H,W] → viridis RGB uint8，供 GIF 帧使用。"""
    t = (np.clip(arr.astype(np.float64), vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    rgba = plt.cm.viridis(t)
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)

EXPERIMENT_ORDER = ("fwi_tv", "fwi_tikhonov", "joint")
EXPERIMENT_LABELS = {
    "fwi_tv": "FWI+TV",
    "fwi_tikhonov": "FWI+Tikhonov",
    "joint": "Joint (Prior)",
}


def load_edm(model_path: str, sampler_config: dict, device: torch.device) -> EDMPrecond:
    if not os.path.isabs(model_path):
        model_path = str(_MANIFOLD_ROOT / model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    sc = dict(sampler_config) if sampler_config else {}
    model = EDMPrecond(
        config=sc,
        img_resolution=sc.get("img_resolution", 70),
        padding_resolution=sc.get("padding_resolution", 72),
        img_channels=sc.get("img_channels", 1),
        label_dim=sc.get("label_dim", 0),
        use_fp16=False,
        sigma_min=sc.get("sigma_min", 0.002),
        sigma_max=sc.get("sigma_max", 80.0),
        sigma_data=sc.get("sigma_data", 0.6),
        model_type=sc.get("model_type", "DhariwalUNet"),
        model_channels=sc.get("model_channels", 32),
        channel_mult=sc.get("channel_mult", [1, 2, 3, 4]),
        channel_mult_emb=sc.get("channel_mult_emb", 4),
        num_blocks=sc.get("num_blocks", 3),
        attn_resolutions=sc.get("attn_resolutions", [32, 16, 8]),
        dropout=sc.get("dropout", 0.0),
        label_dropout=sc.get("label_dropout", 0),
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def sample_ode_latent_grad(
    net: EDMPrecond,
    z: torch.Tensor,
    sigma_max: float,
    num_steps: int,
    sigma_min: float,
    rho: float,
    alpha: float,
    solver: str,
) -> torch.Tensor:
    latents = z * sigma_max
    out = edm_sampler_ode_latentgrad(
        net=net,
        latents=latents,
        class_labels=None,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        alpha=alpha,
        solver=solver,
    )
    while out.dim() > 2:
        out = out.squeeze(0)
    return out


def run_traditional_fwi(
    v_init: torch.Tensor,
    wavefield_loss_fn: WavefieldLoss,
    obs_wf_base: torch.Tensor,
    img_res: int,
    fwi_cfg: dict,
    target_batch: torch.Tensor,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, list[float], list[float], list[float]]:
    mode = mode.lower().strip()
    if mode not in ("tv", "tikhonov"):
        raise ValueError(f"mode must be tv|tikhonov, got {mode}")

    iters = max(1, int(fwi_cfg.get("iterations", 300)))
    lr = float(fwi_cfg.get("lr", 0.03))
    reg_lambda = float(fwi_cfg.get("reg_lambda", 0.01))
    loss_scale = float(fwi_cfg.get("loss_scale", 1.0))
    log_every = max(1, int(fwi_cfg.get("log_every", 10)))
    use_cosine = bool(fwi_cfg.get("cosine_annealing", True))
    eta_min = float(fwi_cfg.get("eta_min", 0.0))

    v = torch.nn.Parameter(v_init.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([v], lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=eta_min)
        if use_cosine
        else None
    )

    loss_hist: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []

    reg_fn = total_variation_loss if mode == "tv" else tikhonov_loss

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        sim = forward_velocity_to_wavefield(v, img_res)
        dl = wavefield_loss_fn(sim, obs_wf_base) * loss_scale
        reg = reg_fn(v)
        total = (dl + reg_lambda * reg).sum()
        total.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        loss_hist.append(float(total.detach().item()))
        with torch.no_grad():
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, v)
        mae_hist.append(mae_n)
        ssim_hist.append(ssim_v)
        if it % log_every == 0 or it == iters - 1:
            print(
                f"  [{mode}] {it + 1}/{iters}  total={loss_hist[-1]:.6f}  {fmt_metrics(mae_n, mae_p, ssim_v)}",
                flush=True,
            )

    return v.detach(), loss_hist, mae_hist, ssim_hist


def run_joint_aux_gen(
    v_init: torch.Tensor,
    z_init: torch.Tensor,
    wavefield_loss_fn: WavefieldLoss,
    obs_wf_base: torch.Tensor,
    img_res: int,
    edm: EDMPrecond,
    joint_cfg: dict,
    target_batch: torch.Tensor,
    device: torch.device,
    edm_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    alpha: float,
    solver: str,
) -> dict[str, Any]:
    """
    联合优化 v_aux 和 z。
    损失 = L_wave(v_aux) + λ_coup·MSE(v_gen(z), v_aux)，不含正则化项。
    λ_coup = 0 时退化为普通数据驱动 FWI（不对 z 反传）。

    返回包含所有追踪信息的 dict：
      v, z, v_gen          — 最终结果张量
      loss, data_loss, coupling — 损失历史
      mae_v, ssim_v        — v_aux vs 真值的指标历史
      mae_gen, ssim_gen    — v_gen vs 真值的指标历史（λ_coup=0 时为空列表）
      grad_v, grad_z       — 每步 ||∇v_aux||、||∇z|| 历史
    """
    iters = max(1, int(joint_cfg.get("iterations", 300)))
    lr_v = float(joint_cfg.get("lr_v", 0.03))
    lr_z = float(joint_cfg.get("lr_z", 0.02))
    coupling_lambda = float(joint_cfg.get("coupling_lambda", 0.1))
    loss_scale = float(joint_cfg.get("loss_scale", 1.0))
    log_every = max(1, int(joint_cfg.get("log_every", 10)))
    use_cosine = bool(joint_cfg.get("cosine_annealing", True))
    eta_min = float(joint_cfg.get("eta_min", 0.0))
    snap_every = max(1, int(joint_cfg.get("snap_every", 30)))
    gif_enabled = bool(joint_cfg.get("gif_enabled", True))

    v = torch.nn.Parameter(v_init.clone().detach().requires_grad_(True))
    z = torch.nn.Parameter(z_init.clone().detach().requires_grad_(True))

    if coupling_lambda > 0:
        opt = torch.optim.Adam([{"params": [v], "lr": lr_v}, {"params": [z], "lr": lr_z}])
    else:
        opt = torch.optim.Adam([v], lr=lr_v)

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=eta_min)
        if use_cosine
        else None
    )

    loss_hist: list[float] = []
    data_hist: list[float] = []
    coup_hist: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    mae_hist_gen: list[float] = []
    ssim_hist_gen: list[float] = []
    grad_v_hist: list[float] = []
    grad_z_hist: list[float] = []

    # 演化快照（CPU tensor，(1,1,H,W) 归一化空间）
    snap_v_aux: list[torch.Tensor] = []
    snap_v_gen: list[torch.Tensor] = []
    snap_labels: list[str] = []

    v_gen_final: torch.Tensor | None = None

    # 初始帧（iter 0）：跑一次 ODE 采样得到 z_init 对应的生成速度场
    snap_v_aux.append(v_init.detach().cpu().clone())
    if coupling_lambda > 0 and gif_enabled:
        with torch.no_grad():
            v_gen_init = sample_ode_latent_grad(edm, z, sigma_max, edm_steps, sigma_min, rho, alpha, solver)
        snap_v_gen.append(v_gen_init.view(1, 1, img_res, img_res).cpu().clone())
    else:
        snap_v_gen.append(torch.zeros_like(v_init.cpu()))
    snap_labels.append("iter 0 (init)")

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        sim = forward_velocity_to_wavefield(v, img_res)
        dl = wavefield_loss_fn(sim, obs_wf_base) * loss_scale
        total = dl

        coup_val = 0.0
        v_gen_b: torch.Tensor | None = None
        if coupling_lambda > 0:
            v_gen = sample_ode_latent_grad(edm, z, sigma_max, edm_steps, sigma_min, rho, alpha, solver)
            v_gen_b = v_gen.view(1, 1, img_res, img_res)
            coup = F.mse_loss(v_gen_b, v)
            total = total + coupling_lambda * coup
            coup_val = float(coup.detach().item())

        total.backward()

        # 梯度 norm（在 optimizer.step() 之前捕获）
        gv = v.grad.norm().item() if v.grad is not None else 0.0
        gz = z.grad.norm().item() if (coupling_lambda > 0 and z.grad is not None) else 0.0
        grad_v_hist.append(gv)
        grad_z_hist.append(gz)

        opt.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        loss_hist.append(float(total.detach().item()))
        data_hist.append(float(dl.detach().item()))
        coup_hist.append(coup_val)

        with torch.no_grad():
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, v)
        mae_hist.append(mae_n)
        ssim_hist.append(ssim_v)

        # v_gen 指标（复用已计算的 v_gen_b，backward 后值仍有效）
        if v_gen_b is not None:
            with torch.no_grad():
                mae_n_gen, mae_p_gen, ssim_gen = velocity_mae_ssim(target_batch, v_gen_b.detach())
            mae_hist_gen.append(mae_n_gen)
            ssim_hist_gen.append(ssim_gen)
            v_gen_final = v_gen_b.detach()
        else:
            mae_n_gen, mae_p_gen, ssim_gen = 0.0, 0.0, 0.0

        if it % log_every == 0 or it == iters - 1:
            gen_line = (
                f"\n    v_gen: {fmt_metrics(mae_n_gen, mae_p_gen, ssim_gen)}" if coupling_lambda > 0 else ""
            )
            print(
                f"  [joint] {it + 1}/{iters}  total={loss_hist[-1]:.6f}  "
                f"data={data_hist[-1]:.6f}  coup={coup_val:.6f}"
                f"\n    v_aux: {fmt_metrics(mae_n, mae_p, ssim_v)}"
                f"  grad_v={gv:.4e}  grad_z={gz:.4e}"
                f"{gen_line}",
                flush=True,
            )

        # 演化快照
        if gif_enabled and ((it + 1) % snap_every == 0 or it == iters - 1):
            snap_v_aux.append(v.detach().cpu().clone())
            snap_v_gen.append(
                v_gen_b.detach().cpu().clone() if v_gen_b is not None
                else torch.zeros(1, 1, img_res, img_res)
            )
            snap_labels.append(f"iter {it + 1}")

    return dict(
        v=v.detach(),
        z=z.detach(),
        v_gen=v_gen_final,
        loss=loss_hist,
        data_loss=data_hist,
        coupling=coup_hist,
        mae_v=mae_hist,
        ssim_v=ssim_hist,
        mae_gen=mae_hist_gen,
        ssim_gen=ssim_hist_gen,
        grad_v=grad_v_hist,
        grad_z=grad_z_hist,
        snap_v_aux=snap_v_aux,
        snap_v_gen=snap_v_gen,
        snap_labels=snap_labels,
        gif_enabled=gif_enabled,
    )


def resolve_experiment_order(cfg: dict) -> list[str]:
    ex_block = cfg.get("experiments") or {}
    active = cfg.get("experiments_active")
    if active is not None:
        if not isinstance(active, (list, tuple)) or len(active) == 0:
            raise ValueError("experiments_active must be a non-empty list or omit for default order")
        order = []
        for k in active:
            ks = str(k).strip()
            if ks not in EXPERIMENT_ORDER:
                raise ValueError(f"Unknown experiment {ks!r}; expected one of {EXPERIMENT_ORDER}")
            if ks not in ex_block:
                raise ValueError(f"experiments.{ks} missing in config")
            order.append(ks)
        return order

    order_raw = list(cfg.get("experiments_order") or list(EXPERIMENT_ORDER))
    order = [k for k in order_raw if k in EXPERIMENT_ORDER and k in ex_block]
    if not order:
        order = [k for k in EXPERIMENT_ORDER if k in ex_block]
    if not order:
        raise ValueError("config.experiments must define at least one experiment block")
    return order


# ---------------------------------------------------------------------------
# 可视化辅助函数
# ---------------------------------------------------------------------------

def _imshow(ax, data: np.ndarray, title: str, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_joint_evolution(name: str, jres: dict, exp_dir: Path) -> None:
    """
    优化演化可视化：
      1. 静态子图：2 行（v_aux / v_gen）× N 列（快照），保存为 PNG
      2. GIF：v_aux 演化 + v_gen 演化（各一个文件）
    """
    snap_v_aux = jres.get("snap_v_aux", [])
    snap_v_gen = jres.get("snap_v_gen", [])
    snap_labels = jres.get("snap_labels", [])
    has_gen = jres.get("v_gen") is not None
    n = len(snap_v_aux)
    if n == 0:
        return

    vm = (-1.0, 1.0)

    # --- 1. 静态子图 ---
    nrows = 2 if has_gen else 1
    fig, axes = plt.subplots(nrows, n, figsize=(max(2.4 * n, 6), 2.8 * nrows + 0.6))
    if n == 1:
        axes = np.array(axes).reshape(nrows, 1)
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for j, (v_aux, label) in enumerate(zip(snap_v_aux, snap_labels)):
        ax = axes[0, j]
        ax.imshow(v_aux[0, 0].numpy(), cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        ax.set_title(label, fontsize=7)
        ax.axis("off")
    axes[0, 0].set_ylabel("v_aux", fontsize=8)

    if has_gen:
        for j, v_gen in enumerate(snap_v_gen):
            ax = axes[1, j]
            ax.imshow(v_gen[0, 0].numpy(), cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
            ax.axis("off")
        axes[1, 0].set_ylabel("v_gen", fontsize=8)

    label_str = EXPERIMENT_LABELS.get(name, name)
    plt.suptitle(f"{label_str} — optimization evolution (normalized μ)", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / f"{name}_evolution.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- 2. GIF ---
    if not jres.get("gif_enabled", True):
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio  # type: ignore
        except ImportError:
            print("  [warn] imageio not found, GIF not saved. Install with: pip install imageio", flush=True)
            return

    rgb_aux = [_vel_to_rgb_uint8(f[0, 0].numpy()) for f in snap_v_aux]
    imageio.mimsave(exp_dir / f"{name}_v_aux_evolution.gif", rgb_aux, duration=0.15, loop=0)
    print(f"  GIF ({len(rgb_aux)} frames) → {name}_v_aux_evolution.gif", flush=True)

    if has_gen:
        rgb_gen = [_vel_to_rgb_uint8(f[0, 0].numpy()) for f in snap_v_gen]
        imageio.mimsave(exp_dir / f"{name}_v_gen_evolution.gif", rgb_gen, duration=0.15, loop=0)
        print(f"  GIF ({len(rgb_gen)} frames) → {name}_v_gen_evolution.gif", flush=True)


def save_joint_comparison(
    name: str,
    r: dict,
    true_np: np.ndarray,
    init_np: np.ndarray,
    exp_dir: Path,
    smooth_sigma: float,
) -> None:
    """
    联合实验对比图（2 行 × 5 列）：
      行 0: True μ | Init | v_aux | v_gen | z（优化噪声）
      行 1: 空白   | 误差图 ×3        | z 绝对值图
    """
    v_aux_np = r["tensor"][0, 0].cpu().numpy()
    v_gen_np = r["v_gen"][0, 0].cpu().numpy() if r["v_gen"] is not None else np.zeros_like(true_np)
    z_np = r["z"][0, 0].cpu().numpy()

    mn_v = r["final_metrics"]
    mn_g = r["final_metrics_gen"]

    vm = (-1.0, 1.0)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))

    # 行 0：速度场
    _imshow(axes[0, 0], true_np, "True μ", vmin=vm[0], vmax=vm[1])
    _imshow(axes[0, 1], init_np, f"Init (σ={smooth_sigma:g})", vmin=vm[0], vmax=vm[1])
    _imshow(
        axes[0, 2], v_aux_np,
        f"v_aux\nMAE_n={mn_v[0]:.4f}  SSIM={mn_v[2]:.4f}",
        vmin=vm[0], vmax=vm[1],
    )
    _imshow(
        axes[0, 3], v_gen_np,
        f"v_gen\nMAE_n={mn_g[0]:.4f}  SSIM={mn_g[2]:.4f}" if r["v_gen"] is not None else "v_gen (N/A)",
        vmin=vm[0], vmax=vm[1],
    )
    zabs = np.abs(z_np)
    _imshow(axes[0, 4], z_np, f"z (optimized noise)\n|z| max={zabs.max():.2f}", cmap="RdBu_r")

    # 行 1：误差图
    axes[1, 0].axis("off")
    for col, (pred, label) in enumerate([(init_np, "Init error"), (v_aux_np, "v_aux error"), (v_gen_np, "v_gen error")], start=1):
        err = pred - true_np
        el = max(np.abs(err).max(), 1e-6)
        axes[1, col].imshow(err, cmap="coolwarm", aspect="auto", vmin=-el, vmax=el)
        axes[1, col].set_title(label, fontsize=7)
        axes[1, col].axis("off")
    _imshow(axes[1, 4], zabs, "|z|", cmap="hot")

    label = EXPERIMENT_LABELS.get(name, name)
    plt.suptitle(f"{label} — velocity comparison (normalized μ)", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / f"{name}_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_joint_trajectories(name: str, r: dict, exp_dir: Path) -> None:
    """
    联合实验轨迹图（2×2 四宫格）：
      (0,0) MAE  — v_aux vs v_gen
      (0,1) SSIM — v_aux vs v_gen
      (1,0) Loss — total / data / coupling
      (1,1) Grad norm — ||∇v_aux|| / ||∇z||
    """
    iters = list(range(1, len(r["mae"]) + 1))
    has_gen = len(r["mae_gen"]) > 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    label = EXPERIMENT_LABELS.get(name, name)

    # (0,0) MAE
    ax = axes[0, 0]
    ax.plot(iters, r["mae"], color="steelblue", label="v_aux")
    if has_gen:
        ax.plot(iters, r["mae_gen"], color="darkorange", linestyle="--", label="v_gen")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("MAE vs True")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) SSIM
    ax = axes[0, 1]
    ax.plot(iters, r["ssim"], color="steelblue", label="v_aux")
    if has_gen:
        ax.plot(iters, r["ssim_gen"], color="darkorange", linestyle="--", label="v_gen")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs True")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Loss
    ax = axes[1, 0]
    ax.plot(iters, r["loss"], color="black", label="total")
    ax.plot(iters, r["data_loss"], color="steelblue", linestyle="--", label="data")
    if has_gen:
        ax.plot(iters, r["coupling"], color="darkorange", linestyle=":", label="coupling")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Gradient norms
    ax = axes[1, 1]
    ax.semilogy(iters, r["grad_v"], color="steelblue", label="||∇v_aux||")
    if has_gen and any(g > 0 for g in r["grad_z"]):
        ax.semilogy(iters, r["grad_z"], color="darkorange", linestyle="--", label="||∇z||")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm (log scale)")
    ax.set_title("Gradient norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{label} — optimization trajectories", fontsize=10)
    plt.tight_layout()
    plt.savefig(exp_dir / f"{name}_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_fwi_trajectories(name: str, r: dict, exp_dir: Path) -> None:
    """FWI 实验轨迹图（1×3）：Loss / MAE / SSIM。"""
    iters = list(range(1, len(r["mae"]) + 1))
    label = EXPERIMENT_LABELS.get(name, name)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    axes[0].plot(iters, r["loss"], color="black")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, r["mae"], color="steelblue")
    axes[1].set_title("MAE (normalized) vs True")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, r["ssim"], color="darkorange")
    axes[2].set_title("SSIM vs True")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{label} — optimization trajectory", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / f"{name}_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_overview_comparison(
    order: list[str],
    results: dict,
    true_np: np.ndarray,
    init_np: np.ndarray,
    smooth_sigma: float,
    exp_dir: Path,
) -> None:
    """所有实验的速度场纵览对比图（含 True + Init 列）。"""
    vm = (-1.0, 1.0)

    # 列：True/Init + 每个 FWI 实验的 v_aux + joint 实验的 v_aux 和 v_gen
    cols: list[tuple[np.ndarray, str]] = [
        (true_np, "True μ"),
        (init_np, f"Init (σ={smooth_sigma:g})"),
    ]
    for name in order:
        r = results[name]
        pred = r["tensor"][0, 0].cpu().numpy()
        mn = r["final_metrics"]
        cols.append((pred, f"{EXPERIMENT_LABELS.get(name, name)}\nMAE_n={mn[0]:.4f} SSIM={mn[2]:.4f}"))
        if r["kind"] == "joint" and r.get("v_gen") is not None:
            v_gen_np = r["v_gen"][0, 0].cpu().numpy()
            mn_g = r["final_metrics_gen"]
            cols.append((v_gen_np, f"v_gen ({name})\nMAE_n={mn_g[0]:.4f} SSIM={mn_g[2]:.4f}"))

    ncols = len(cols)
    fig, axes = plt.subplots(2, ncols, figsize=(2.8 * ncols + 0.5, 5.5))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for j, (img, title) in enumerate(cols):
        axes[0, j].imshow(img, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[0, j].set_title(title, fontsize=7)
        axes[0, j].axis("off")
        if j == 0:
            axes[1, j].axis("off")
        else:
            err = img - true_np
            el = max(np.abs(err).max(), 1e-6)
            axes[1, j].imshow(err, cmap="coolwarm", aspect="auto", vmin=-el, vmax=el)
            axes[1, j].set_title("error", fontsize=7)
            axes[1, j].axis("off")

    plt.suptitle("Overview: all experiments (normalized velocity μ)", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "comparison_overview.png", dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is required for wavefield forward modeling.")

    parser = argparse.ArgumentParser(description="Aux velocity + EDM joint optimization vs FWI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_SCRIPT_DIR / "config_aux_vel_edm_joint_opt.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        cp.cuda.Device(device.index if device.index is not None else 0).use()

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = cfg.get("target", {}).get("data_path") or str(_MANIFOLD_ROOT / "data" / "Curvevel-B")
    file_index = int(cfg.get("target", {}).get("file_index", 60))
    sample_index = int(cfg.get("target", {}).get("sample_index", 0))

    sampler_config = cfg.get("sampler", {})
    img_res = int(sampler_config.get("img_resolution", 70))

    init_cfg = cfg.get("init") or {}
    smooth_sigma = float(init_cfg.get("smooth_sigma", 10.0))

    fwi_common = cfg.get("fwi_common") or {}
    fwi_loss_type = str(fwi_common.get("loss_type", "mse")).lower().strip()
    if fwi_loss_type not in FWI_LOSS_TYPES:
        raise ValueError(f"fwi_common.loss_type must be one of {FWI_LOSS_TYPES}")

    w2_global = cfg.get("w2_per_trace") or {}
    wavefield_loss_fn = build_wavefield_loss(fwi_loss_type, w2_global)

    observed = load_observed_wavefield(data_path, file_index, sample_index, device)  # (5, 1000, 70)
    target = load_dataset_sample(data_path, file_index, sample_index, device)  # (70, 70)
    target_batch = target.unsqueeze(0).unsqueeze(0)  # (1, 1, 70, 70)

    from scipy.ndimage import gaussian_filter

    t_np = target_batch[0, 0].detach().cpu().numpy()
    x0_smooth = (
        torch.from_numpy(gaussian_filter(t_np, sigma=smooth_sigma))
        .float()
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    model_path = cfg["model"]["model_path"]
    if not os.path.isabs(model_path):
        model_path = str(_MANIFOLD_ROOT / model_path)
    edm = load_edm(model_path, sampler_config, device)

    es = cfg.get("edm_sampler") or {}
    edm_num_steps = int(es.get("num_steps", 20))
    edm_sigma_min = float(es.get("sigma_min", 0.002))
    edm_sigma_max = float(es.get("sigma_max", 80.0))
    edm_rho = float(es.get("rho", 7.0))
    edm_alpha = float(es.get("alpha", 1.0))
    edm_solver = str(es.get("solver", "heun")).lower()

    joint_z_cfg = cfg.get("joint_z") or {}
    z_seed = int(joint_z_cfg.get("seed", seed + 9000))
    g_z = torch.Generator(device=device).manual_seed(z_seed)
    z_init = torch.randn(1, 1, img_res, img_res, device=device, dtype=torch.float32, generator=g_z)

    ex_block = cfg.get("experiments") or {}
    order = resolve_experiment_order(cfg)

    print(
        f"\n=== aux_vel_edm_joint_opt  |  file={file_index} sample={sample_index}  "
        f"smooth_σ={smooth_sigma}  order={order} ===\n",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}

    for name in order:
        ecfg = ex_block.get(name)
        if not ecfg:
            raise ValueError(f"experiments.{name} missing")
        exp_type = str(ecfg.get("type", name)).lower().strip()
        mcfg = {**fwi_common, **ecfg}

        if exp_type in ("fwi_tv", "tv"):
            print(f"--- {EXPERIMENT_LABELS.get(name, name)} (traditional TV) ---", flush=True)
            v_out, lh, mh, sh = run_traditional_fwi(
                x0_smooth, wavefield_loss_fn, observed, img_res, mcfg, target_batch, device, "tv"
            )
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, v_out)
            results[name] = {
                "kind": "fwi",
                "tensor": v_out,
                "loss": lh,
                "mae": mh,
                "ssim": sh,
                "final_metrics": (mae_n, mae_p, ssim_v),
            }

        elif exp_type in ("fwi_tikhonov", "tikhonov"):
            print(f"--- {EXPERIMENT_LABELS.get(name, name)} (traditional Tikhonov) ---", flush=True)
            v_out, lh, mh, sh = run_traditional_fwi(
                x0_smooth, wavefield_loss_fn, observed, img_res, mcfg, target_batch, device, "tikhonov"
            )
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, v_out)
            results[name] = {
                "kind": "fwi",
                "tensor": v_out,
                "loss": lh,
                "mae": mh,
                "ssim": sh,
                "final_metrics": (mae_n, mae_p, ssim_v),
            }

        elif exp_type == "joint":
            print(f"--- {EXPERIMENT_LABELS.get(name, name)} (v_aux + z, prior coupling) ---", flush=True)
            jres = run_joint_aux_gen(
                x0_smooth,
                z_init,
                wavefield_loss_fn,
                observed,
                img_res,
                edm,
                mcfg,
                target_batch,
                device,
                edm_num_steps,
                edm_sigma_min,
                edm_sigma_max,
                edm_rho,
                edm_alpha,
                edm_solver,
            )
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, jres["v"])
            final_metrics_gen = (0.0, 0.0, 0.0)
            if jres["v_gen"] is not None:
                final_metrics_gen = velocity_mae_ssim(target_batch, jres["v_gen"])
            results[name] = {
                "kind": "joint",
                "tensor": jres["v"],
                "v_gen": jres["v_gen"],
                "z": jres["z"],
                "loss": jres["loss"],
                "data_loss": jres["data_loss"],
                "coupling": jres["coupling"],
                "mae": jres["mae_v"],
                "ssim": jres["ssim_v"],
                "mae_gen": jres["mae_gen"],
                "ssim_gen": jres["ssim_gen"],
                "grad_v": jres["grad_v"],
                "grad_z": jres["grad_z"],
                "snap_v_aux": jres["snap_v_aux"],
                "snap_v_gen": jres["snap_v_gen"],
                "snap_labels": jres["snap_labels"],
                "gif_enabled": jres["gif_enabled"],
                "final_metrics": (mae_n, mae_p, ssim_v),
                "final_metrics_gen": final_metrics_gen,
            }

        else:
            raise ValueError(f"Unknown experiment type {exp_type!r} for {name}")

        mn = results[name]["final_metrics"]
        print(f"  v_aux final vs true: {fmt_metrics(mn[0], mn[1], mn[2])}", flush=True)
        if results[name]["kind"] == "joint" and results[name]["v_gen"] is not None:
            mn_g = results[name]["final_metrics_gen"]
            print(f"  v_gen final vs true: {fmt_metrics(mn_g[0], mn_g[1], mn_g[2])}", flush=True)
        print(flush=True)

    # -----------------------------------------------------------------------
    # 保存输出
    # -----------------------------------------------------------------------
    outdir = cfg.get("paths", {}).get("outdir", "demo_output")
    if not os.path.isabs(outdir):
        outdir = str(_SCRIPT_DIR / outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(outdir) / f"aux_vel_edm_joint_f{file_index}_s{sample_index}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, exp_dir / "config.yaml")

    true_np = t_np  # 归一化，已在前面计算
    init_np = x0_smooth[0, 0].cpu().numpy()

    # metrics.csv
    with open(exp_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "var", "mae_norm", "mae_phys_m_s", "ssim"])
        for name in order:
            r = results[name]
            mn = r["final_metrics"]
            w.writerow([name, "v_aux", mn[0], mn[1], mn[2]])
            if r["kind"] == "joint":
                mn_g = r.get("final_metrics_gen", (0, 0, 0))
                w.writerow([name, "v_gen", mn_g[0], mn_g[1], mn_g[2]])

    # 各实验可视化
    for name in order:
        r = results[name]
        if r["kind"] == "fwi":
            save_fwi_trajectories(name, r, exp_dir)
        elif r["kind"] == "joint":
            save_joint_comparison(name, r, true_np, init_np, exp_dir, smooth_sigma)
            save_joint_trajectories(name, r, exp_dir)
            save_joint_evolution(name, r, exp_dir)

    # 总览对比图
    save_overview_comparison(order, results, true_np, init_np, smooth_sigma, exp_dir)

    # summary.json
    summary: dict[str, Any] = {
        "exp_dir": str(exp_dir),
        "file_index": file_index,
        "sample_index": sample_index,
        "order": order,
        "metrics": {},
    }
    for name in order:
        r = results[name]
        entry: dict[str, Any] = {"v_aux": list(r["final_metrics"])}
        if r["kind"] == "joint":
            entry["v_gen"] = list(r.get("final_metrics_gen", (0, 0, 0)))
        summary["metrics"][name] = entry
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
