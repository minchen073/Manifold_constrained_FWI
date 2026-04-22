"""
辅助变量联合优化 v2 — 解决 v1 中两变量互相拉扯的问题。

与 v1 的关键区别：
1. **z 初始化**：对光滑速度场添加中间时刻噪声后运行正向 ODE，得到接近光滑速度场的初始噪声，
   而非随机初始化。
2. **交替优化**：v_phys 和 z 使用独立优化器，每步轮流更新，不再联合反传。
3. **v_phys 引导项**：波场误差梯度 + RED-diff 范式的 L2 引导（朝向 v_gen），与 RED-diffeq
   实验中的正则化强度同一量级。
4. **z 目标**：每步最小化 MSE(v_gen(z), v_phys.detach())，使先验速度场始终追踪物理速度场。

运行（Manifold_constrained_FWI 目录）:
  uv run python exp/aux_vel_edm_joint_opt_v2/demo_aux_vel_edm_joint_opt_v2.py \\
    --config exp/aux_vel_edm_joint_opt_v2/config_aux_vel_edm_joint_opt_v2.yaml
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
from typing import Any, Optional

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
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
from src.core.generate import edm_sampler_ode, edm_sampler_ode_latentgrad, forward_edm_sampler
from src.core.loss import WavefieldLoss
from src.seismic import seismic_master_forward_modeling

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def v_denormalize(v_norm: torch.Tensor) -> torch.Tensor:
    return v_norm * 1500.0 + 3000.0


def velocity_mae_ssim(target: torch.Tensor, pred: torch.Tensor):
    t = target.detach().float()
    p = pred.detach().float()
    for x in (t, p):
        pass
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
    data = np.load(data_file)
    target = data[sample_index]
    if target.ndim == 3:
        target = target[0]
    elif target.ndim == 4:
        target = target[0, 0]
    target_norm = (target.astype(np.float32) - 3000.0) / 1500.0
    return torch.from_numpy(target_norm).float().to(device)


def load_observed_wavefield(data_path: str, file_index: int, sample_index: int, device):
    data_file = Path(data_path) / f"data{file_index}.npy"
    if not data_file.exists():
        raise FileNotFoundError(f"Wavefield data not found: {data_file}")
    data = np.load(data_file)
    return torch.from_numpy(data[sample_index].astype(np.float32)).float().to(device)


def forward_velocity_to_wavefield(v_norm: torch.Tensor, img_res: int) -> torch.Tensor:
    velocity_physical = v_denormalize(v_norm).clamp(1500.0, 4500.0)
    velocity_2d = velocity_physical.squeeze().reshape(img_res, img_res).to(v_norm.device)
    sim = seismic_master_forward_modeling(velocity_2d)
    return sim


def build_wavefield_loss(loss_type: str, w2_cfg: dict) -> WavefieldLoss:
    lt = str(loss_type).lower().strip()
    dt = 0.001
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


# ---------------------------------------------------------------------------
# z 初始化：光滑速度场 → 中间时刻跳跃 → 正向 ODE → z_init
# ---------------------------------------------------------------------------

def init_z_from_smooth_velocity(
    edm: EDMPrecond,
    v_smooth: torch.Tensor,
    sigma_mid: float,
    sigma_max: float,
    num_steps: int,
    rho: float,
    alpha: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将光滑速度场通过中间时刻噪声跳跃 + 正向 ODE 转换为初始噪声 z_init（单位尺度）。

    步骤：
      1. v_smooth + sigma_mid * eps  →  在中间时刻的加噪速度
      2. forward_edm_sampler(sigma_min=sigma_mid, sigma_max=sigma_max)  →  z_scaled
      3. z_init = z_scaled / sigma_max  →  单位尺度（与优化变量 z 同尺度）

    这样 z_init → v_gen = ODE(z_init * sigma_max) ≈ v_smooth，
    解决了随机初始化时先验速度场与物理速度场 gap 过大的问题。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(v_smooth.shape, device=device, dtype=v_smooth.dtype, generator=g)
    v_noisy = v_smooth + float(sigma_mid) * eps

    with torch.no_grad():
        z_scaled = forward_edm_sampler(
            edm,
            v_noisy,
            class_labels=None,
            num_steps=num_steps,
            sigma_min=float(sigma_mid),
            sigma_max=float(sigma_max),
            rho=float(rho),
            alpha=float(alpha),
            solver="heun",
        )
    return z_scaled / float(sigma_max)


# ---------------------------------------------------------------------------
# 生成工具
# ---------------------------------------------------------------------------

def generate_no_grad(
    edm: EDMPrecond,
    z: torch.Tensor,
    sigma_max: float,
    sigma_min: float,
    num_steps: int,
    rho: float,
    alpha: float,
    solver: str,
) -> torch.Tensor:
    """从 z（单位尺度）生成速度场，不保留梯度（用于 v_phys 引导项）。"""
    latents = z.detach() * float(sigma_max)
    with torch.no_grad():
        out = edm_sampler_ode(
            edm,
            latents,
            class_labels=None,
            num_steps=num_steps,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            rho=float(rho),
            alpha=float(alpha),
            solver=solver,
        )
    return out


def generate_with_grad(
    edm: EDMPrecond,
    z: torch.Tensor,
    sigma_max: float,
    sigma_min: float,
    num_steps: int,
    rho: float,
    alpha: float,
    solver: str,
) -> torch.Tensor:
    """从 z（单位尺度）生成速度场，保留对 z 的梯度（用于 z 更新）。"""
    latents = z * float(sigma_max)
    return edm_sampler_ode_latentgrad(
        edm,
        latents,
        class_labels=None,
        num_steps=num_steps,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        rho=float(rho),
        alpha=float(alpha),
        solver=solver,
    )


# ---------------------------------------------------------------------------
# 主优化循环
# ---------------------------------------------------------------------------

def run_alternating_joint_opt(
    v_init: torch.Tensor,
    z_init: torch.Tensor,
    wavefield_loss_fn: WavefieldLoss,
    obs_wf: torch.Tensor,
    img_res: int,
    edm: EDMPrecond,
    cfg: dict,
    target_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """
    交替优化 v_phys 和 z：

    每步：
      (A) v_phys 更新：
          - v_gen = ODE(z.detach())  （无梯度，质量优先）
          - data_loss.backward()
          - v_phys.grad += reg_lambda * (v_phys - v_gen) / numel  （RED-diff 风格引导）
          - opt_v.step()

      (B) z 更新：
          - v_gen_grad = ODE_latentgrad(z)  （有梯度）
          - coupling_loss = MSE(v_gen_grad, v_phys.detach())
          - coupling_loss.backward()
          - opt_z.step()
    """
    iters = max(1, int(cfg.get("iterations", 300)))
    lr_v = float(cfg.get("lr_v", 0.03))
    lr_z = float(cfg.get("lr_z", 0.01))
    reg_lambda = float(cfg.get("reg_lambda", 0.5))
    loss_scale = float(cfg.get("loss_scale", 1.0))
    log_every = max(1, int(cfg.get("log_every", 10)))
    snap_every = max(1, int(cfg.get("snap_every", 30)))
    gif_enabled = bool(cfg.get("gif_enabled", True))
    use_cosine = bool(cfg.get("cosine_annealing", True))
    eta_min = float(cfg.get("eta_min", 0.0))

    sigma_max = float(cfg.get("sigma_max", 80.0))
    sigma_min = float(cfg.get("sigma_min", 0.002))
    rho = float(cfg.get("rho", 7.0))
    alpha = float(cfg.get("alpha", 1.0))

    # v_phys 更新时用的 ODE（无梯度，可用更多步）
    edm_v_steps = int(cfg.get("edm_v_num_steps", 10))
    edm_v_solver = str(cfg.get("edm_v_solver", "euler")).lower()

    # z 更新时用的 ODE（有梯度，步数少以节省显存）
    edm_z_steps = int(cfg.get("edm_z_num_steps", 5))
    edm_z_solver = str(cfg.get("edm_z_solver", "euler")).lower()

    v = torch.nn.Parameter(v_init.clone().detach().requires_grad_(True))
    z = torch.nn.Parameter(z_init.clone().detach().requires_grad_(True))

    opt_v = torch.optim.Adam([v], lr=lr_v)
    opt_z = torch.optim.Adam([z], lr=lr_z)

    sched_v = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=iters, eta_min=eta_min)
        if use_cosine else None
    )
    sched_z = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_z, T_max=iters, eta_min=eta_min)
        if use_cosine else None
    )

    # 历史记录
    data_loss_hist: list[float] = []
    coupling_hist: list[float] = []
    mae_v_hist: list[float] = []
    ssim_v_hist: list[float] = []
    mae_gen_hist: list[float] = []
    ssim_gen_hist: list[float] = []
    grad_v_hist: list[float] = []
    grad_z_hist: list[float] = []

    # 快照（CPU）
    snap_v: list[torch.Tensor] = []
    snap_gen: list[torch.Tensor] = []
    snap_labels: list[str] = []

    # 初始帧
    snap_v.append(v_init.detach().cpu().clone())
    with torch.no_grad():
        v_gen_init = generate_no_grad(edm, z, sigma_max, sigma_min, edm_v_steps, rho, alpha, edm_v_solver)
    snap_gen.append(v_gen_init.detach().cpu().clone())
    snap_labels.append("iter 0 (init)")

    v_gen_current: torch.Tensor = v_gen_init.detach()

    for it in range(iters):
        # ============================================================
        # (A) 更新 v_phys
        # ============================================================
        # 先生成 v_gen（不需要 z 的梯度）
        v_gen_current = generate_no_grad(
            edm, z, sigma_max, sigma_min, edm_v_steps, rho, alpha, edm_v_solver
        )

        opt_v.zero_grad(set_to_none=True)
        sim = forward_velocity_to_wavefield(v, img_res)
        data_loss = wavefield_loss_fn(sim, obs_wf) * loss_scale
        data_loss.backward()

        # RED-diff 风格引导：将 v_gen 视为去噪后的先验速度场
        with torch.no_grad():
            if v.grad is not None:
                guidance = v.detach() - v_gen_current.reshape_as(v.detach())
                scale = reg_lambda / v.numel()
                v.grad.add_(guidance, alpha=scale)

        opt_v.step()
        if sched_v is not None:
            sched_v.step()
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        gv = v.grad.norm().item() if v.grad is not None else 0.0
        grad_v_hist.append(gv)
        data_loss_hist.append(float(data_loss.detach().item()))

        # ============================================================
        # (B) 更新 z：最小化 MSE(v_gen(z), v_phys.detach())
        # ============================================================
        opt_z.zero_grad(set_to_none=True)
        v_gen_grad = generate_with_grad(
            edm, z, sigma_max, sigma_min, edm_z_steps, rho, alpha, edm_z_solver
        )
        coupling_loss = F.mse_loss(v_gen_grad.reshape_as(v.detach()), v.detach())
        coupling_loss.backward()
        opt_z.step()

        gz = z.grad.norm().item() if z.grad is not None else 0.0
        grad_z_hist.append(gz)
        coupling_hist.append(float(coupling_loss.detach().item()))

        # ============================================================
        # 指标
        # ============================================================
        with torch.no_grad():
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, v)
            mae_n_gen, mae_p_gen, ssim_gen = velocity_mae_ssim(target_batch, v_gen_current.reshape_as(v))

        mae_v_hist.append(mae_n)
        ssim_v_hist.append(ssim_v)
        mae_gen_hist.append(mae_n_gen)
        ssim_gen_hist.append(ssim_gen)

        if it % log_every == 0 or it == iters - 1:
            print(
                f"  [v2] {it + 1}/{iters}  data={data_loss_hist[-1]:.6f}  coup={coupling_hist[-1]:.6f}"
                f"\n    v_phys: {fmt_metrics(mae_n, mae_p, ssim_v)}  ||∇v||={gv:.3e}  ||∇z||={gz:.3e}"
                f"\n    v_gen:  {fmt_metrics(mae_n_gen, mae_p_gen, ssim_gen)}",
                flush=True,
            )

        # 快照
        if gif_enabled and ((it + 1) % snap_every == 0 or it == iters - 1):
            snap_v.append(v.detach().cpu().clone())
            snap_gen.append(v_gen_current.reshape_as(v).detach().cpu().clone())
            snap_labels.append(f"iter {it + 1}")

    with torch.no_grad():
        v_gen_final = generate_no_grad(
            edm, z, sigma_max, sigma_min, edm_v_steps, rho, alpha, edm_v_solver
        )

    return dict(
        v=v.detach(),
        z=z.detach(),
        v_gen=v_gen_final.reshape_as(v).detach(),
        data_loss=data_loss_hist,
        coupling=coupling_hist,
        mae_v=mae_v_hist,
        ssim_v=ssim_v_hist,
        mae_gen=mae_gen_hist,
        ssim_gen=ssim_gen_hist,
        grad_v=grad_v_hist,
        grad_z=grad_z_hist,
        snap_v=snap_v,
        snap_gen=snap_gen,
        snap_labels=snap_labels,
        gif_enabled=gif_enabled,
    )


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _vel_to_rgb_uint8(arr: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    t = (np.clip(arr.astype(np.float64), vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    rgba = plt.cm.viridis(t)
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def save_comparison(r: dict, true_np: np.ndarray, init_np: np.ndarray, exp_dir: Path) -> None:
    v_np = r["v"][0, 0].cpu().numpy()
    vg_np = r["v_gen"][0, 0].cpu().numpy()
    vm = (-1.0, 1.0)

    mn_v = r["final_metrics_v"]
    mn_g = r["final_metrics_gen"]

    fig, axes = plt.subplots(2, 5, figsize=(16, 5.5))

    panels_top = [
        (true_np, "True"),
        (init_np, "Init (smooth)"),
        (v_np, f"v_phys\nMAE_n={mn_v[0]:.4f} SSIM={mn_v[2]:.4f}"),
        (vg_np, f"v_gen\nMAE_n={mn_g[0]:.4f} SSIM={mn_g[2]:.4f}"),
    ]
    for j, (arr, title) in enumerate(panels_top):
        axes[0, j].imshow(arr, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[0, j].set_title(title, fontsize=7)
        axes[0, j].axis("off")

    # z noise
    z_np = r["z"][0, 0].cpu().numpy()
    axes[0, 4].imshow(z_np, cmap="RdBu_r", aspect="auto")
    axes[0, 4].set_title(f"z (optimized)\n|z|_max={np.abs(z_np).max():.2f}", fontsize=7)
    axes[0, 4].axis("off")

    # Error maps
    axes[1, 0].axis("off")
    for j, (pred, label) in enumerate(
        [(init_np, "Init error"), (v_np, "v_phys error"), (vg_np, "v_gen error")], start=1
    ):
        err = pred - true_np
        el = max(np.abs(err).max(), 1e-6)
        axes[1, j].imshow(err, cmap="coolwarm", aspect="auto", vmin=-el, vmax=el)
        axes[1, j].set_title(label, fontsize=7)
        axes[1, j].axis("off")
    axes[1, 4].imshow(np.abs(z_np), cmap="hot", aspect="auto")
    axes[1, 4].set_title("|z|", fontsize=7)
    axes[1, 4].axis("off")

    plt.suptitle("aux_vel_edm_joint_opt_v2 — velocity comparison", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_trajectories(r: dict, exp_dir: Path) -> None:
    iters = list(range(1, len(r["mae_v"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(iters, r["mae_v"], color="steelblue", label="v_phys")
    axes[0, 0].plot(iters, r["mae_gen"], color="darkorange", linestyle="--", label="v_gen")
    axes[0, 0].set_title("MAE vs True")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, r["ssim_v"], color="steelblue", label="v_phys")
    axes[0, 1].plot(iters, r["ssim_gen"], color="darkorange", linestyle="--", label="v_gen")
    axes[0, 1].set_title("SSIM vs True")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, r["data_loss"], color="black", label="data loss")
    axes[1, 0].plot(iters, r["coupling"], color="darkorange", linestyle="--", label="coupling (z)")
    axes[1, 0].set_title("Loss curves")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(iters, r["grad_v"], color="steelblue", label="||∇v||")
    axes[1, 1].semilogy(iters, r["grad_z"], color="darkorange", linestyle="--", label="||∇z||")
    axes[1, 1].set_title("Gradient norms")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("aux_vel_edm_joint_opt_v2 — optimization trajectories", fontsize=10)
    plt.tight_layout()
    plt.savefig(exp_dir / "trajectories.png", dpi=160, bbox_inches="tight")
    plt.close()


def _snap_to_arr(t: torch.Tensor) -> np.ndarray:
    """将快照张量转为 (H, W) numpy，兼容 4D/3D/2D。"""
    if t.dim() == 4:
        return t[0, 0].numpy()
    if t.dim() == 3:
        return t[0].numpy()
    return t.numpy()


def save_evolution_panels(r: dict, true_np: np.ndarray, exp_dir: Path) -> None:
    """两行子图：上行 v_phys，下行 v_gen，每列为一个优化快照。
    第一列额外展示真实速度场作为参考。"""
    snaps_v = r.get("snap_v", [])
    snaps_gen = r.get("snap_gen", [])
    labels = r.get("snap_labels", [])
    n = len(snaps_v)
    if n == 0:
        return

    vm = (-1.0, 1.0)
    ncols = n + 1  # +1 for true reference
    fig, axes = plt.subplots(2, ncols, figsize=(2.5 * ncols, 5.2))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    # 第 0 列：真实速度场（参考）
    for row in range(2):
        im = axes[row, 0].imshow(true_np, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)
    axes[0, 0].set_title("True", fontsize=8)
    axes[1, 0].set_title("True (ref)", fontsize=8)

    for j in range(n):
        col = j + 1
        label = labels[j] if j < len(labels) else f"snap {j}"

        # 上行：v_phys
        v_arr = _snap_to_arr(snaps_v[j])
        im0 = axes[0, col].imshow(v_arr, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[0, col].set_title(label, fontsize=7)
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        # 下行：v_gen
        g_arr = _snap_to_arr(snaps_gen[j]) if j < len(snaps_gen) else np.zeros_like(v_arr)
        im1 = axes[1, col].imshow(g_arr, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

    # 行标签
    axes[0, 0].set_ylabel("v_phys →", fontsize=9, labelpad=4)
    axes[1, 0].set_ylabel("v_gen →", fontsize=9, labelpad=4)

    plt.suptitle("Optimization evolution: v_phys (top row) vs v_gen (bottom row)", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "evolution_panels.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  evolution_panels.png saved ({ncols} columns)", flush=True)


def save_evolution_gif(r: dict, exp_dir: Path) -> None:
    if not r.get("gif_enabled", True):
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio  # type: ignore
        except ImportError:
            print("  [warn] imageio not found, skip GIF", flush=True)
            return

    snaps_v = r.get("snap_v", [])
    snaps_gen = r.get("snap_gen", [])
    if not snaps_v:
        return

    rgb_v = [_vel_to_rgb_uint8(_snap_to_arr(f)) for f in snaps_v]
    imageio.mimsave(exp_dir / "v_phys_evolution.gif", rgb_v, duration=0.15, loop=0)
    print(f"  GIF: {len(rgb_v)} frames → v_phys_evolution.gif", flush=True)

    if snaps_gen:
        rgb_gen = [_vel_to_rgb_uint8(_snap_to_arr(f)) for f in snaps_gen]
        imageio.mimsave(exp_dir / "v_gen_evolution.gif", rgb_gen, duration=0.15, loop=0)
        print(f"  GIF: {len(rgb_gen)} frames → v_gen_evolution.gif", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is required for wavefield forward modeling.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(_SCRIPT_DIR / "config_aux_vel_edm_joint_opt_v2.yaml"),
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

    sampler_cfg = cfg.get("sampler", {})
    img_res = int(sampler_cfg.get("img_resolution", 70))

    smooth_sigma = float((cfg.get("init") or {}).get("smooth_sigma", 10.0))
    fwi_loss_type = str((cfg.get("fwi_common") or {}).get("loss_type", "mse")).lower().strip()
    w2_cfg = cfg.get("w2_per_trace") or {}
    wavefield_loss_fn = build_wavefield_loss(fwi_loss_type, w2_cfg)

    observed = load_observed_wavefield(data_path, file_index, sample_index, device)
    target = load_dataset_sample(data_path, file_index, sample_index, device)
    target_batch = target.unsqueeze(0).unsqueeze(0)

    from scipy.ndimage import gaussian_filter

    t_np = target_batch[0, 0].detach().cpu().numpy()
    v_smooth = (
        torch.from_numpy(gaussian_filter(t_np, sigma=smooth_sigma))
        .float()
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    edm = load_edm(cfg["model"]["model_path"], sampler_cfg, device)

    # === z 初始化 ===
    z_init_cfg = cfg.get("z_init") or {}
    sigma_mid = float(z_init_cfg.get("sigma_mid", 5.0))
    sigma_max = float(sampler_cfg.get("sigma_max", 80.0))
    sigma_min_net = float(sampler_cfg.get("sigma_min", 0.002))
    z_init_steps = int(z_init_cfg.get("num_steps", 20))
    z_init_rho = float(z_init_cfg.get("rho", 7.0))
    z_init_alpha = float(z_init_cfg.get("alpha", 1.0))
    z_init_seed = int(z_init_cfg.get("seed", seed + 7777))

    print(f"\n=== Initializing z from smooth velocity (sigma_mid={sigma_mid}) ===", flush=True)
    z_init = init_z_from_smooth_velocity(
        edm, v_smooth, sigma_mid, sigma_max, z_init_steps,
        z_init_rho, z_init_alpha, z_init_seed, device,
    )
    z_init = z_init.reshape(1, 1, img_res, img_res)

    # 验证初始化质量
    with torch.no_grad():
        v_gen_check = generate_no_grad(
            edm, z_init, sigma_max, sigma_min_net,
            int((cfg.get("opt") or {}).get("edm_v_num_steps", 10)), 7.0, 1.0, "euler",
        ).reshape(1, 1, img_res, img_res)
    mae_init, _, ssim_init = velocity_mae_ssim(target_batch, v_gen_check)
    mae_smooth, _, ssim_smooth = velocity_mae_ssim(target_batch, v_smooth)
    print(f"  z_init → v_gen vs true:   MAE_n={mae_init:.4f} SSIM={ssim_init:.4f}", flush=True)
    print(f"  v_smooth vs true:         MAE_n={mae_smooth:.4f} SSIM={ssim_smooth:.4f}", flush=True)

    # === 交替优化 ===
    opt_cfg = cfg.get("opt") or {}
    opt_cfg_full = {
        **opt_cfg,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min_net,
        "rho": float(sampler_cfg.get("rho", 7.0)),
        "alpha": float(sampler_cfg.get("alpha", 1.0)),
    }

    print(
        f"\n=== Alternating joint optimization  |  "
        f"file={file_index} sample={sample_index}  iters={opt_cfg.get('iterations', 300)} ===\n",
        flush=True,
    )
    results = run_alternating_joint_opt(
        v_smooth, z_init, wavefield_loss_fn, observed,
        img_res, edm, opt_cfg_full, target_batch, device,
    )

    final_metrics_v = velocity_mae_ssim(target_batch, results["v"].reshape(1, 1, img_res, img_res))
    final_metrics_gen = velocity_mae_ssim(target_batch, results["v_gen"].reshape(1, 1, img_res, img_res))
    results["final_metrics_v"] = final_metrics_v
    results["final_metrics_gen"] = final_metrics_gen
    results["v"] = results["v"].reshape(1, 1, img_res, img_res)
    results["v_gen"] = results["v_gen"].reshape(1, 1, img_res, img_res)
    results["z"] = results["z"].reshape(1, 1, img_res, img_res)

    print(f"\n  v_phys final: {fmt_metrics(*final_metrics_v)}", flush=True)
    print(f"  v_gen  final: {fmt_metrics(*final_metrics_gen)}", flush=True)

    # === 保存输出 ===
    outdir = str(_SCRIPT_DIR / cfg.get("paths", {}).get("outdir", "demo_output"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(outdir) / f"v2_f{file_index}_s{sample_index}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, exp_dir / "config.yaml")

    true_np = t_np
    init_np = v_smooth[0, 0].cpu().numpy()

    save_comparison(results, true_np, init_np, exp_dir)
    save_trajectories(results, exp_dir)
    save_evolution_panels(results, true_np, exp_dir)
    save_evolution_gif(results, exp_dir)

    # numpy 保存
    np.save(exp_dir / "v_phys_final.npy", results["v"][0, 0].cpu().numpy())
    np.save(exp_dir / "v_gen_final.npy", results["v_gen"][0, 0].cpu().numpy())
    np.save(exp_dir / "z_final.npy", results["z"][0, 0].cpu().numpy())
    np.save(exp_dir / "z_init.npy", z_init[0, 0].cpu().numpy())

    # summary
    summary = {
        "file_index": file_index,
        "sample_index": sample_index,
        "z_init_sigma_mid": sigma_mid,
        "final_metrics_v": list(final_metrics_v),
        "final_metrics_gen": list(final_metrics_gen),
        "v_smooth_metrics": list(velocity_mae_ssim(target_batch, v_smooth)),
        "z_init_gen_metrics": [mae_init, 0.0, ssim_init],
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
