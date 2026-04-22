#!/usr/bin/env python3
"""
mid_time_opt：中间时刻参数化 FWI

优化变量：z（与速度场同形状）
速度场生成：v_gen = D_θ(v_base + σ·z, σ)  （单步去噪器调用，无 ODE）
损失：wave_MSE(F(v_gen), d_obs) + λ·R(z)
  R(z)：chi 分布负对数似然，λ=0 则关闭
v_base：真解的高斯光滑化，固定不优化
z 初始化：投影初始化——v_base + σ·ε →[逆向 ODE σ→0]→ v_proj + σ·ε₂ → z_init

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/mid_time_opt/demo_mid_time_opt.py \\
    --config exp/mid_time_opt/config_mid_time_opt.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
from src.core.generate import edm_sampler_ode
from src.seismic import seismic_master_forward_modeling


# ===========================================================================
# Utilities
# ===========================================================================

def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (_MANIFOLD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def v_normalize(v_phys: np.ndarray) -> np.ndarray:
    return (v_phys - 3000.0) / 1500.0


def v_denormalize_np(v_norm) -> np.ndarray:
    if isinstance(v_norm, torch.Tensor):
        x = v_norm.detach().float().cpu().numpy()
    else:
        x = np.asarray(v_norm, dtype=np.float32)
    return (x * 1500.0 + 3000.0).squeeze()


def v_denormalize_tensor(v_norm: torch.Tensor) -> torch.Tensor:
    return v_norm * 1500.0 + 3000.0


def velocity_mae_ssim(
    pred_np: np.ndarray, target_np: np.ndarray,
    device: torch.device, vmin: float, vmax: float,
) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred_np - target_np)))
    def to_t(a: np.ndarray) -> torch.Tensor:
        a = np.squeeze(a)
        t = torch.from_numpy(a.astype(np.float32)).view(1, 1, a.shape[0], a.shape[1])
        return ((t - vmin) / (vmax - vmin)).clamp(0.0, 1.0).to(device)
    ssim_val = float(
        pytorch_ssim.ssim(to_t(pred_np), to_t(target_np), window_size=11, size_average=True).item()
    )
    return mae, ssim_val


def load_target_velocity(npy_path: Path, sample_index: int) -> np.ndarray:
    data = np.load(npy_path)
    return data[sample_index, 0].astype(np.float32)


def build_edm_precond(sc: dict, device: torch.device) -> EDMPrecond:
    model = EDMPrecond(
        config=sc,
        img_resolution=sc["img_resolution"],
        padding_resolution=sc["padding_resolution"],
        img_channels=sc["img_channels"],
        label_dim=sc.get("label_dim", 0),
        use_fp16=False,
        sigma_min=sc["sigma_min"],
        sigma_max=sc["sigma_max"],
        sigma_data=sc["sigma_data"],
        model_type=sc.get("model_type", "DhariwalUNet"),
        model_channels=sc["model_channels"],
        channel_mult=sc["channel_mult"],
        channel_mult_emb=sc["channel_mult_emb"],
        num_blocks=sc["num_blocks"],
        attn_resolutions=sc["attn_resolutions"],
        dropout=sc["dropout"],
        label_dropout=sc["label_dropout"],
    ).to(device)
    train_cfg = sc.get("train") or {}
    if train_cfg.get("load_model", True):
        model_path = _resolve_path(
            train_cfg.get("model_path") or "pretrained_model/Curvevel-B_EDM/trained_model.pth"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"EDM model not found: {model_path}")
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    return model


# ===========================================================================
# 核心：单步去噪生成速度场
# ===========================================================================

def denoise_step(net: EDMPrecond, v_base: torch.Tensor, z: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    单步去噪：v_gen = D_θ(v_base + σ·z, σ)

    v_base: (1,1,H,W) 归一化速度场，固定
    z:      (1,1,H,W) 优化变量，初始化为 N(0,I)
    sigma:  标量，中间时刻噪声水平

    返回 v_gen: (1,1,H,W) 归一化速度场，梯度流经 z
    """
    x_noisy = v_base + sigma * z
    sigma_t = torch.full((x_noisy.shape[0],), sigma, device=x_noisy.device, dtype=torch.float32)
    v_gen = net(x_noisy, sigma_t, None)
    return v_gen


def forward_wave(v_gen: torch.Tensor) -> torch.Tensor:
    v_phys = v_denormalize_tensor(v_gen.squeeze()).clamp(1500.0, 4500.0)
    return seismic_master_forward_modeling(v_phys)


# ===========================================================================
# Chi-NLL 正则化
# ===========================================================================

def chi_nll_reg(z: torch.Tensor) -> torch.Tensor:
    """
    chi 分布负对数似然（归一化常数已减去，最小值约为 0）：
      R(z) = -((d-1)/2) * log(||z||²) + ||z||²/2 - C
    鼓励 z 落在半径 √(d-1) 的球面附近（标准高斯的典型集）。
    """
    d = z.numel()
    r_sq = torch.sum(z.view(-1) ** 2)
    r_star_sq = float(d - 1)
    reg_min = 0.5 * r_star_sq * (1.0 - math.log(r_star_sq + 1e-8))
    return -0.5 * (d - 1) * torch.log(r_sq + 1e-8) + 0.5 * r_sq - reg_min


# ===========================================================================
# z 投影初始化
# ===========================================================================

def init_z_by_projection(
    net: EDMPrecond,
    v_base: torch.Tensor,
    sigma: float,
    sigma_min: float,
    num_steps: int,
    rho: float,
    alpha: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    投影初始化：v_base + σ·ε₁ →[逆向 ODE σ→0]→ v_proj + σ·ε₂ → z_init

    流程：
      1. x_noisy = v_base + σ·ε₁          （加噪到优化 σ 水平）
      2. v_proj  = ODE(x_noisy, σ→0)       （逆向 ODE 投影到流形）
      3. z_init  = (v_proj + σ·ε₂ - v_base) / σ  （重新加噪，得到 z_init）

    使得 D_θ(v_base + σ·z_init, σ) ≈ v_proj（流形投影速度场）。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    eps1 = torch.randn(v_base.shape, device=device, dtype=v_base.dtype, generator=g)
    x_noisy = (v_base + sigma * eps1).detach()

    with torch.no_grad():
        v_proj = edm_sampler_ode(
            net=net,
            latents=x_noisy,
            class_labels=None,
            num_steps=num_steps,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma),   # ODE 从 σ 积分到 0
            rho=float(rho),
            alpha=float(alpha),
            solver="heun",
        )

    eps2 = torch.randn(v_base.shape, device=device, dtype=v_base.dtype, generator=g)
    x_reproj = v_proj + sigma * eps2
    z_init = (x_reproj - v_base) / sigma   # = ε₂ + (v_proj − v_base)/σ
    return z_init.detach()


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="中间时刻参数化 FWI：优化 z 最小化 D_θ(v+σz,σ) 的波场 MSE")
    parser.add_argument(
        "--config", type=str,
        default=str(_SCRIPT_DIR / "config_mid_time_opt.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path.resolve())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str)
    cp.cuda.Device(device.index if device.index is not None else 0).use()

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 模型
    scfg = cfg["sampler"]
    net = build_edm_precond(scfg, device)
    net.eval()

    # 参数
    sigma      = float((cfg.get("denoise_step") or {}).get("sigma", 1.0))
    opt_cfg    = cfg.get("optimization") or {}
    opt_steps  = int(opt_cfg.get("opt_steps", 200))
    lr         = float(opt_cfg.get("lr", 0.01))
    snapshots  = int(opt_cfg.get("snapshots", 10))
    reg_weight = float(opt_cfg.get("reg_weight", 0.0))
    vb_cfg     = cfg.get("v_base") or {}
    smooth_sigma = float(vb_cfg.get("smooth_sigma", 10.0))
    viz        = cfg.get("visualization") or {}
    vel_vmin   = float(viz.get("vel_vmin_m_s", 1500.0))
    vel_vmax   = float(viz.get("vel_vmax_m_s", 4500.0))
    _wave_shot = int(viz.get("wave_plot_shot", 0))  # reserved, not used in multi-stage

    # 数据
    cv_cfg       = cfg.get("curvevel") or {}
    model60_path = _resolve_path(cv_cfg["model60_path"])
    sample_index = int(cv_cfg.get("sample_index", 0))
    if not model60_path.is_file():
        raise SystemExit(f"Data not found: {model60_path}")

    target_vel_np = load_target_velocity(model60_path, sample_index)
    with torch.no_grad():
        target_wave = seismic_master_forward_modeling(
            torch.from_numpy(target_vel_np).float().to(device)
        )

    # 输出目录
    outdir_rel = (cfg.get("paths") or {}).get("outdir", "demo_output")
    out_base = Path(outdir_rel)
    if not out_base.is_absolute():
        out_base = _SCRIPT_DIR / out_base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base.resolve() / f"mid_time_opt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out_dir / "config.yaml")
    print(f"\nOutput: {out_dir}", flush=True)

    # -----------------------------------------------------------------------
    # v_base：真解的光滑化（固定）
    # -----------------------------------------------------------------------
    v_smooth_phys = gaussian_filter(target_vel_np, sigma=smooth_sigma)
    v_smooth_norm = v_normalize(v_smooth_phys)
    v_base = (
        torch.from_numpy(v_smooth_norm).float().to(device)
        .unsqueeze(0).unsqueeze(0)
    )  # (1,1,H,W), 不参与优化

    mae_smooth, ssim_smooth = velocity_mae_ssim(v_smooth_phys, target_vel_np, device, vel_vmin, vel_vmax)
    print(
        f"  v_base (smooth σ={smooth_sigma}) vs true: MAE={mae_smooth:.1f} m/s  SSIM={ssim_smooth:.4f}",
        flush=True,
    )
    # -----------------------------------------------------------------------
    # 多阶段参数
    # -----------------------------------------------------------------------
    zinit_cfg       = cfg.get("z_init") or {}
    zinit_steps     = int(zinit_cfg.get("num_steps", 20))
    zinit_rho       = float(zinit_cfg.get("rho", 7.0))
    zinit_alpha     = float(zinit_cfg.get("alpha", 1.0))
    gen_ode_steps   = int(zinit_cfg.get("gen_steps", 20))
    sigma_min_model = float(scfg.get("sigma_min", 0.002))
    num_stages      = int(opt_cfg.get("num_stages", 3))

    # 跨阶段记录
    all_vproj_np:    list[np.ndarray]   = []   # 每阶段 ODE 生成速度场
    all_mae_proj:    list[float]        = []
    all_ssim_proj:   list[float]        = []
    stage_summaries: list[dict]         = []

    # -----------------------------------------------------------------------
    # 多阶段优化主循环
    # -----------------------------------------------------------------------
    for stage in range(num_stages):
        print(
            f"\n{'='*60}\n  Stage {stage}  σ={sigma}  opt_steps={opt_steps}  lr={lr}\n{'='*60}",
            flush=True,
        )
        v_base_np_stage = v_denormalize_np(v_base)  # 当前阶段 v_base（物理单位）

        # -------------------------------------------------------------------
        # z 初始化：第 0 阶段投影初始化，后续阶段随机高斯
        # -------------------------------------------------------------------
        if stage == 0:
            print(
                f"  z init: ODE projection  num_steps={zinit_steps}  σ: {sigma:.3f}→0",
                flush=True,
            )
            z_tensor = init_z_by_projection(
                net, v_base, sigma, sigma_min_model,
                zinit_steps, zinit_rho, zinit_alpha, seed, device,
            )
            z_init_label = "z init\n(projection)"
        else:
            stage_seed = seed + stage
            g = torch.Generator(device=device).manual_seed(stage_seed)
            z_tensor = torch.randn(v_base.shape, device=device, generator=g)
            print(f"  z init: random Gaussian  seed={stage_seed}", flush=True)
            z_init_label = f"z init\n(rand seed={stage_seed})"

        z = torch.nn.Parameter(z_tensor, requires_grad=True)
        z_init_np = z.detach().cpu().numpy().squeeze().copy()

        # 记录初始生成质量
        with torch.no_grad():
            v_gen_init_np = v_denormalize_np(denoise_step(net, v_base, z, sigma))
        mae_init, ssim_init = velocity_mae_ssim(v_gen_init_np, target_vel_np, device, vel_vmin, vel_vmax)
        print(
            f"  D_θ(v_base + σ·z_init, σ) vs true: MAE={mae_init:.1f} m/s  SSIM={ssim_init:.4f}",
            flush=True,
        )

        # -------------------------------------------------------------------
        # 优化循环
        # -------------------------------------------------------------------
        opt_adam = torch.optim.Adam([z], lr=lr)
        snap_times = np.clip(
            np.unique(np.round(np.linspace(0, opt_steps, snapshots, endpoint=True)).astype(int)),
            0, opt_steps,
        )

        hist_vel:    list[np.ndarray] = []
        hist_z:      list[np.ndarray] = []
        hist_labels: list[str]        = []

        def capture(label: str, _v_base=v_base, _z=z) -> None:
            with torch.no_grad():
                vg = denoise_step(net, _v_base, _z, sigma)
            hist_vel.append(v_denormalize_np(vg).copy())
            hist_z.append(_z.detach().cpu().numpy().squeeze().copy())
            hist_labels.append(label)

        losses:      list[float] = []
        wave_losses: list[float] = []
        reg_losses:  list[float] = []
        mae_hist:    list[float] = []
        ssim_hist:   list[float] = []
        grad_hist:   list[float] = []

        if 0 in snap_times:
            capture("iter 0")

        for it in range(opt_steps):
            opt_adam.zero_grad(set_to_none=True)
            v_gen = denoise_step(net, v_base, z, sigma)
            pred_wave = forward_wave(v_gen)
            wave_mse = F.mse_loss(pred_wave, target_wave)

            if reg_weight > 0.0:
                reg = chi_nll_reg(z)
                loss = wave_mse + reg_weight * reg
                reg_val = float((reg_weight * reg).item())
            else:
                loss = wave_mse
                reg_val = 0.0

            loss.backward()
            grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
            opt_adam.step()

            losses.append(loss.item())
            wave_losses.append(wave_mse.item())
            reg_losses.append(reg_val)
            grad_hist.append(grad_norm)

            with torch.no_grad():
                pv = v_denormalize_np(v_gen)
                mae_v, ssim_v = velocity_mae_ssim(pv, target_vel_np, device, vel_vmin, vel_vmax)
            mae_hist.append(mae_v)
            ssim_hist.append(ssim_v)

            if (it + 1) in snap_times:
                capture(f"iter {it + 1}")

            if (it + 1) % max(1, opt_steps // 10) == 0 or it == 0:
                reg_str = f"  reg={reg_val:.6f}" if reg_weight > 0.0 else ""
                print(
                    f"  iter {it+1}/{opt_steps}  wave MSE={wave_mse.item():.6f}{reg_str}  "
                    f"MAE={mae_v:.1f} m/s  SSIM={ssim_v:.4f}  ||∇z||={grad_norm:.4e}",
                    flush=True,
                )

        # -------------------------------------------------------------------
        # 阶段结束：以 v_base + σ·z 为 σ 时刻含噪变量，ODE 生成干净样本
        # -------------------------------------------------------------------
        x_sigma = (v_base + sigma * z.detach()).detach()
        print(
            f"\n  Stage {stage} → ODE generation: x_σ = v_base + σ·z  "
            f"(σ={sigma:.3f}→0, steps={gen_ode_steps})",
            flush=True,
        )
        with torch.no_grad():
            v_proj = edm_sampler_ode(
                net=net,
                latents=x_sigma,
                class_labels=None,
                num_steps=gen_ode_steps,
                sigma_min=float(sigma_min_model),
                sigma_max=float(sigma),
                rho=float(zinit_rho),
                alpha=float(zinit_alpha),
                solver="heun",
            )
        v_proj_np = v_denormalize_np(v_proj)
        mae_proj, ssim_proj = velocity_mae_ssim(v_proj_np, target_vel_np, device, vel_vmin, vel_vmax)
        print(
            f"  ODE sample vs true: MAE={mae_proj:.1f} m/s  SSIM={ssim_proj:.4f}",
            flush=True,
        )
        all_vproj_np.append(v_proj_np)
        all_mae_proj.append(mae_proj)
        all_ssim_proj.append(ssim_proj)

        # -------------------------------------------------------------------
        # 图 A：阶段演化快照（2 行：v_gen 快照 / z 快照）
        # -------------------------------------------------------------------
        ncols = 2 + len(hist_vel)
        z_lim = max(3.0,
                    float(np.abs(z_init_np).max()) + 0.1,
                    max((float(np.abs(hz).max()) for hz in hist_z), default=0.0) + 0.1)
        fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5.5))
        axes = np.atleast_2d(axes)

        axes[0, 0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, 0].set_title("True v", fontsize=8)
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

        vb_title = f"v_base\n(smooth σ={smooth_sigma})" if stage == 0 else f"v_base\n(stage {stage-1} ODE)"
        axes[0, 1].imshow(v_base_np_stage, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes[0, 1].set_title(vb_title, fontsize=8)
        axes[0, 1].axis("off")
        axes[1, 1].imshow(z_init_np, cmap="coolwarm", aspect="auto", vmin=-z_lim, vmax=z_lim)
        axes[1, 1].set_title(z_init_label, fontsize=8)
        axes[1, 1].axis("off")

        for j, (vel, zj, lbl) in enumerate(zip(hist_vel, hist_z, hist_labels)):
            col = j + 2
            axes[0, col].imshow(vel, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
            axes[0, col].set_title(lbl, fontsize=8)
            axes[0, col].axis("off")
            axes[1, col].imshow(zj, cmap="coolwarm", aspect="auto", vmin=-z_lim, vmax=z_lim)
            axes[1, col].axis("off")

        plt.suptitle(
            f"Stage {stage}  σ={sigma}  reg_weight={reg_weight}\n"
            f"v_gen = D_θ(v_base + σ·z, σ)  |  z optimized via Adam lr={lr}",
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"evolution_stage{stage}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # -------------------------------------------------------------------
        # 图 B：阶段 ODE 生成样本
        # -------------------------------------------------------------------
        err_proj = v_proj_np - target_vel_np
        err_lim_proj = float(np.max(np.abs(err_proj))) + 1e-6
        fig, axes_p = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, (arr, title, cmap, vm0, vm1) in zip(axes_p, [
            (target_vel_np,    "True v (m/s)",    "viridis",  vel_vmin,       vel_vmax),
            (v_base_np_stage,  "v_base (m/s)",    "viridis",  vel_vmin,       vel_vmax),
            (v_proj_np,        "ODE sample (m/s)","viridis",  vel_vmin,       vel_vmax),
            (err_proj,         "ODE − true (m/s)","coolwarm", -err_lim_proj,  err_lim_proj),
        ]):
            im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vm0, vmax=vm1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.suptitle(
            f"Stage {stage} ODE sample  (x_σ=v_base+σ·z → ODE {gen_ode_steps} steps, σ={sigma:.3f}→0)\n"
            f"MAE={mae_proj:.1f} m/s  SSIM={ssim_proj:.4f}",
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"ode_sample_stage{stage}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # -------------------------------------------------------------------
        # 图 C：阶段指标曲线
        # -------------------------------------------------------------------
        use_reg = reg_weight > 0.0
        iters = list(range(1, len(losses) + 1))
        def _setup_ax(ax, title, ylabel=None, xlabel=None, log=False):
            ax.set_title(title)
            if ylabel:
                ax.set_ylabel(ylabel)
            if xlabel:
                ax.set_xlabel(xlabel)
            if log:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

        if use_reg:
            fig, axes_m = plt.subplots(3, 2, figsize=(10, 10))
            axes_m[0, 0].plot(iters, wave_losses, color="C0")
            _setup_ax(axes_m[0, 0], "Wave MSE", ylabel="loss")
            axes_m[0, 1].plot(iters, reg_losses, color="C5")
            _setup_ax(axes_m[0, 1], f"Reg loss (λ={reg_weight})", ylabel="loss")
            axes_m[1, 0].plot(iters, losses, color="C0", label="total")
            axes_m[1, 0].plot(iters, wave_losses, "C0--", alpha=0.5, label="wave")
            axes_m[1, 0].plot(iters, reg_losses,  "C5--", alpha=0.5, label="reg")
            axes_m[1, 0].legend(fontsize=8)
            _setup_ax(axes_m[1, 0], "Total loss")
            axes_m[1, 1].plot(iters, mae_hist, color="C1")
            _setup_ax(axes_m[1, 1], "Velocity MAE (m/s)", ylabel="MAE")
            axes_m[2, 0].plot(iters, ssim_hist, color="C2")
            _setup_ax(axes_m[2, 0], "Velocity SSIM", ylabel="SSIM", xlabel="iter")
            axes_m[2, 1].semilogy(iters, grad_hist, color="C3")
            _setup_ax(axes_m[2, 1], "||∇z|| (log)", ylabel="grad norm", xlabel="iter")
        else:
            fig, axes_m = plt.subplots(2, 2, figsize=(9, 5.5))
            axes_m[0, 0].plot(iters, wave_losses, color="C0")
            _setup_ax(axes_m[0, 0], "Wave MSE", ylabel="loss")
            axes_m[0, 1].plot(iters, mae_hist, color="C1")
            _setup_ax(axes_m[0, 1], "Velocity MAE (m/s)", ylabel="MAE")
            axes_m[1, 0].plot(iters, ssim_hist, color="C2")
            _setup_ax(axes_m[1, 0], "Velocity SSIM", ylabel="SSIM", xlabel="iter")
            axes_m[1, 1].semilogy(iters, grad_hist, color="C3")
            _setup_ax(axes_m[1, 1], "||∇z|| (log)", ylabel="grad norm", xlabel="iter")
        plt.suptitle(
            f"Stage {stage} metrics  σ={sigma}  lr={lr}  reg_weight={reg_weight}", fontsize=10
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"metrics_stage{stage}.png", dpi=150, bbox_inches="tight")
        plt.close()

        stage_summaries.append({
            "stage": stage,
            "z_init": {"mae_m_s": mae_init, "ssim": ssim_init},
            "opt_final": {
                "wave_mse": wave_losses[-1],
                "mae_m_s": mae_hist[-1],
                "ssim": ssim_hist[-1],
            },
            "ode_sample": {"mae_m_s": mae_proj, "ssim": ssim_proj},
        })

        # -------------------------------------------------------------------
        # 更新 v_base：ODE 生成样本作为下一阶段的 v_base
        # -------------------------------------------------------------------
        v_base = v_proj.detach()

    # -----------------------------------------------------------------------
    # 图：多阶段 ODE 样本总览
    # -----------------------------------------------------------------------
    ncols_s = 1 + len(all_vproj_np)
    fig, axes_s = plt.subplots(1, ncols_s, figsize=(3.5 * ncols_s, 4))
    axes_s = np.atleast_1d(axes_s)
    axes_s[0].imshow(target_vel_np, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
    axes_s[0].set_title("True v", fontsize=9)
    axes_s[0].axis("off")
    for i, vp in enumerate(all_vproj_np):
        im = axes_s[i + 1].imshow(vp, cmap="viridis", aspect="auto", vmin=vel_vmin, vmax=vel_vmax)
        axes_s[i + 1].set_title(
            f"Stage {i} ODE\nMAE={all_mae_proj[i]:.0f}  SSIM={all_ssim_proj[i]:.3f}", fontsize=8
        )
        axes_s[i + 1].axis("off")
        plt.colorbar(im, ax=axes_s[i + 1], fraction=0.046)
    plt.suptitle(
        f"Multi-stage mid_time_opt  σ={sigma}  {num_stages} stages  opt_steps={opt_steps}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "stages_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "sigma": sigma,
        "num_stages": num_stages,
        "opt_steps": opt_steps,
        "lr": lr,
        "reg_weight": reg_weight,
        "v_base_smooth": {"smooth_sigma": smooth_sigma, "mae_m_s": mae_smooth, "ssim": ssim_smooth},
        "stages": stage_summaries,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done. Output: {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
