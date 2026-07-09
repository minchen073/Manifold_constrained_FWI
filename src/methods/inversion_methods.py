"""统一接口的 FWI 反演方法集合。

包含 6 个方法的 ``run_*`` 函数：

  ┌─────────────────┬──────────────────────────────────────────────────────────┐
  │  Method         │  Reference & key idea                                     │
  ├─────────────────┼──────────────────────────────────────────────────────────┤
  │  Tikhonov       │  L2 平滑：R = ‖∇μ‖²                                       │
  │  TV             │  Total Variation：R = ‖∇μ‖₁                              │
  │  RED-DiffEq     │  Shan et al. 2026, arXiv:2509.21659                      │
  │                 │  ∇R = E_{t,ε}[ε_θ(x_t,t) − ε]，物理空间优化              │
  │  DiffusionFWI   │  Wang et al. 2023                                        │
  │                 │  反向扩散链中嵌入 K 步 FWI 梯度                          │
  │  DiffusionILVR  │  Taufik et al.；ILVR (Choi et al. ICCV'21) + DiffusionFWI │
  │                 │  在 DiffusionFWI 基础上每步做低频替换                    │
  │  Method B       │  本仓库（联合交替优化）                                  │
  │                 │  双变量 v + z：v 物理空间 + L2 manifold 引导，z 跟踪 v   │
  └─────────────────┴──────────────────────────────────────────────────────────┘

公共接口
========

每个 ``run_*`` 函数签名一致：

    run_xxx(
        seismic_obs:    torch.Tensor,           # (S, T, R) 观测波场
        velocity_true:  torch.Tensor,           # (H, W) ground truth m/s
        forward_fn:     Callable,               # v_phys(70,70) m/s → seismic
        diffusion:      Optional[DiffusionPrior],  # 仅扩散类方法需要
        device:         torch.device,
        params:         dict,                   # 方法专属超参（见 default_params）
    ) → InversionResult

返回 ``InversionResult`` ：

    velocity_pred:  np.ndarray  (H, W) 预测速度场（m/s）
    velocity_init:  np.ndarray  (H, W) 初始速度场（m/s）
    history: dict  {
        'rmse': List[float],     # 每步 RMSE（归一化 [-1,1] 域）
        'mae':  List[float],     # 每步 MAE（归一化 [-1,1] 域）
        'ssim': List[float],     # 每步 SSIM（归一化到 [0,1]）
        'obs_loss':   List[float],
        'reg_loss':   List[float],   # 没有正则项时全 0
        'total_loss': List[float],
    }
    extra: dict   方法专属附加信息
    method: str   方法名
    params: dict  生效的超参（含默认值合并后的结果）

约定
====

* 所有标量损失/指标按 *归一化 [-1,1] 速度域* 计算（与 RED-DiffEq 论文一致），
  方便和 paper Table 直接对比。
* 速度物理范围 :math:`[1500, 4500]` m/s 被仿射归一化到 :math:`[-1, 1]`：
    :math:`x = (v - 3000) / 1500`
* 缺省超参取 RED-DiffEq 论文 SI Sec.5 (OpenFWI 设置)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from src.core import pytorch_ssim


# =============================================================================
# 归一化常数（OpenFWI / RED-DiffEq 共用）
# =============================================================================
VELOCITY_CENTER_M_S: float = 3000.0
VELOCITY_SCALE_M_S: float = 1500.0
VELOCITY_VMIN_M_S: float = 1500.0
VELOCITY_VMAX_M_S: float = 4500.0


def _v_to_norm(v_phys: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """物理 m/s → [-1, 1]."""
    return (v_phys - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S


def _v_to_phys(v_norm: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """[-1, 1] → 物理 m/s."""
    return v_norm * VELOCITY_SCALE_M_S + VELOCITY_CENTER_M_S


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class DiffusionPrior:
    """扩散先验包装：DDPM 网络 + DDPM 噪声调度表（alphas_cumprod）。

    与 ``OpenFWIUNetWrapper``/``DDPMScheduler`` 兼容；调用 ``denoise(x_t, t)``
    会得到模型直接预测的噪声 :math:`\\epsilon_\\theta(x_t,t)`。
    """

    wrapper: torch.nn.Module
    alphas_cumprod: torch.Tensor          # (T,) DDPM 累积 alpha
    num_train_timesteps: int = 1000
    final_alpha_cumprod: float = 1.0      # DDIM 边界，t=-1 处的 ᾱ，用于反向最末步

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """ε_θ(x_t, t)。``t`` 可为 int 或 (B,) long tensor。"""
        if isinstance(t, int):
            t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        else:
            t_tensor = t.to(device=x_t.device, dtype=torch.long)
        return self.wrapper(x_t, t_tensor).sample

    def alpha_bar(self, t: int) -> torch.Tensor:
        """ᾱ_t (scalar)；t==-1 → final_alpha_cumprod。"""
        if t < 0:
            return torch.tensor(self.final_alpha_cumprod, dtype=torch.float32)
        return self.alphas_cumprod[int(t)].float()


@dataclass
class InversionResult:
    velocity_pred: np.ndarray
    velocity_init: np.ndarray
    history: Dict[str, List[float]]
    extra: Dict = field(default_factory=dict)
    method: str = ""
    params: Dict = field(default_factory=dict)


# =============================================================================
# 通用工具
# =============================================================================
_DEFAULT_OBS_LOSS = "l1"


def _obs_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = _DEFAULT_OBS_LOSS,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """观测数据拟合 loss。

    ``mask``（可选，1=observed, 0=missing）：用于缺失迹场景，与 red-diffeq
    ``core/losses.py`` 一致 —— 仅在 observed 位置上做平均（``sum / mask.sum()``）。
    缺省 ``mask=None`` 退化为对所有 (S,T,R) 元素的常规均值。
    """
    if kind == "l1":
        per = (pred - target).abs()
    elif kind in ("l2", "mse"):
        per = (pred - target).pow(2)
    else:
        raise ValueError(f"unknown obs loss: {kind}")
    if mask is None:
        return per.mean()
    m = mask.to(dtype=per.dtype, device=per.device)
    return (per * m).sum() / m.sum().clamp(min=1.0)


def apply_missing_traces(
    seismic: torch.Tensor,
    num_missing: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """随机置零 ``num_missing`` 个 receiver 道，返回 (masked_seismic, mask)。

    输入支持 (S, T, R) 或 (B, S, T, R)。**所有 source 共享同一组缺失 receiver
    索引**（论文 Sec 2.3.3：active receiver locations remain identical for every
    shot of a given velocity model）。

    Args:
      seismic: 观测波场张量
      num_missing: 缺失道数 (0 ≤ num_missing ≤ R)
      seed: 若非 None，固定 receiver 索引选取（用于 resume / 方法间一致）

    Returns:
      (masked_seismic, mask)：mask 同形 1=observed / 0=missing，dtype 与
      ``seismic`` 一致；num_missing=0 时返回原 tensor + 全 1 mask。
    """
    if num_missing < 0:
        raise ValueError(f"num_missing must be >= 0, got {num_missing}")
    n_receivers = seismic.shape[-1]
    if num_missing > n_receivers:
        raise ValueError(
            f"num_missing ({num_missing}) exceeds receiver count ({n_receivers})"
        )
    mask = torch.ones_like(seismic)
    if num_missing == 0:
        return seismic, mask
    if seed is not None:
        gen = torch.Generator(device=seismic.device).manual_seed(int(seed))
        idx = torch.randperm(n_receivers, generator=gen, device=seismic.device)[:num_missing]
    else:
        idx = torch.randperm(n_receivers, device=seismic.device)[:num_missing]
    masked = seismic.clone()
    masked[..., idx] = 0
    mask[..., idx] = 0
    return masked, mask


def _smoothed_init_norm(v_true_phys: np.ndarray, sigma: float) -> np.ndarray:
    """高斯平滑 ground-truth 作初始模型，返回 [-1, 1] 域。"""
    v_norm = (v_true_phys - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S
    return gaussian_filter(v_norm, sigma=sigma).astype(np.float32)


def _to_norm_4d(v: torch.Tensor) -> torch.Tensor:
    """squeezed → (1,1,H,W)."""
    if v.ndim == 2:
        return v.unsqueeze(0).unsqueeze(0)
    if v.ndim == 3:
        return v.unsqueeze(0)
    return v


def _ssim_norm(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """对 [-1,1] 速度计算 SSIM；先线性 0-1 化再喂给 pytorch_ssim。"""
    a = _to_norm_4d(pred_norm).float().clamp(-1.0, 1.0)
    b = _to_norm_4d(target_norm).float().clamp(-1.0, 1.0)
    a01 = (a + 1.0) * 0.5
    b01 = (b + 1.0) * 0.5
    return pytorch_ssim.ssim(a01, b01, window_size=11, size_average=True)


def _metrics_norm(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> Tuple[float, float, float]:
    """归一化 [-1,1] 速度域上的 (rmse, mae, ssim)。"""
    diff = (pred_norm.float() - target_norm.float()).flatten()
    rmse = float(diff.pow(2).mean().sqrt().item())
    mae = float(diff.abs().mean().item())
    ssim = float(_ssim_norm(pred_norm, target_norm).item())
    return rmse, mae, ssim


def _q_sample(x0: torch.Tensor, t: int, alphas_cumprod: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """前向扩散 :math:`x_t = \\sqrt{\\bar\\alpha_t}\\, x_0 + \\sqrt{1-\\bar\\alpha_t}\\, \\epsilon`."""
    abar = alphas_cumprod[int(t)].to(x0.device).float()
    return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise


def _predict_x0_from_eps(
    x_t: torch.Tensor, eps: torch.Tensor, t: int, alphas_cumprod: torch.Tensor, clip: bool = True
) -> torch.Tensor:
    """从噪声预测干净样本 :math:`\\hat x_0`."""
    abar = alphas_cumprod[int(t)].to(x_t.device).float()
    x0 = (x_t - (1.0 - abar).sqrt() * eps) / abar.sqrt()
    return x0.clamp(-1.0, 1.0) if clip else x0


def _rederive_eps(
    x_t: torch.Tensor, x0: torch.Tensor, t: int, alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    abar = alphas_cumprod[int(t)].to(x_t.device).float()
    return (x_t - abar.sqrt() * x0) / (1.0 - abar).sqrt()


def _q_posterior_mean(
    x0: torch.Tensor, x_t: torch.Tensor, t: int, alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    """DDPM 反向后验均值 :math:`\\mu(x_t, x_0, t)`。"""
    abar_t = alphas_cumprod[int(t)].to(x_t.device).float()
    abar_prev = (alphas_cumprod[int(t) - 1].to(x_t.device).float() if t > 0
                 else torch.tensor(1.0, device=x_t.device, dtype=torch.float32))
    beta_t = 1.0 - abar_t / abar_prev
    coef_x0 = beta_t * abar_prev.sqrt() / (1.0 - abar_t)
    coef_xt = (1.0 - abar_prev) * (1.0 - beta_t).sqrt() / (1.0 - abar_t)
    return coef_x0 * x0 + coef_xt * x_t


def _ddpm_posterior_variance(t: int, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """DDPM 反向后验方差 σ_t² = β_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t)。

    对齐 ``diffusers.DDPMScheduler._get_variance`` 的 ``fixed_small`` 选项
    （DDPM 原文 Ho et al. 2020 推荐，diffusers 默认）。完全由 β schedule
    决定，无可调系数。
    """
    device = alphas_cumprod.device
    abar_t = alphas_cumprod[int(t)].to(device).float()
    abar_prev = (alphas_cumprod[int(t) - 1].to(device).float() if t > 0
                 else torch.tensor(1.0, device=device, dtype=torch.float32))
    beta_t = 1.0 - abar_t / abar_prev
    var = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
    return var.clamp(min=1e-20)


def _ddpm_step(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t: int,
    alphas_cumprod: torch.Tensor,
    *,
    clip_x0: bool = True,
    noise_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """标准 DDPM 反向单步 (x_t, ε̂_θ) → x_{t-1}。对齐 ``diffusers.DDPMScheduler.step``。

    步骤（Ho et al. 2020 / Wang 2023 eq.6）：
      1. x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t        （可选 clip 到 [−1, 1]）
      2. μ_t = posterior_mean(x̂₀, x_t, t)
      3. 当 t > 0 时加噪：x_{t−1} = μ_t + noise_scale · σ_t · z, z ∼ 𝒩(0, I)
         其中 σ_t² 由 β schedule 决定（fixed_small）；t == 0 时直接取 μ。

    ``noise_scale``：随机项缩放系数。1.0 = 标准 DDPM 祖先采样；0.001 = 对齐
    Taufik ilvrefwi/diffefwi ``p_sample_wf`` 的近确定性采样（FWI 主循环默认）；
    0.0 = 纯 μ_t（完全确定性）。

    返回 ``(x_prev, x0_pred)``。
    """
    x0_pred = _predict_x0_from_eps(x_t, eps_pred, t, alphas_cumprod, clip=clip_x0)
    mean = _q_posterior_mean(x0_pred, x_t, int(t), alphas_cumprod)
    if t > 0 and noise_scale != 0.0:
        var = _ddpm_posterior_variance(int(t), alphas_cumprod)
        if generator is not None:
            noise = torch.randn(
                x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype,
            )
        else:
            noise = torch.randn_like(x_t)
        x_prev = mean + noise_scale * var.sqrt() * noise
    else:
        x_prev = mean
    return x_prev, x0_pred


def _low_pass_avgpool(x: torch.Tensor, factor: int) -> torch.Tensor:
    """ILVR 低通滤波：avg_pool ↓N + 最近邻 ↑N。``factor=1`` 时为恒等。"""
    if factor <= 1:
        return x
    h, w = x.shape[-2:]
    down = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    up = F.interpolate(down, size=(h, w), mode="nearest")
    return up


def _gaussian_kernel(sigma: float, ksize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1 通道 2D 高斯核 (1, 1, k, k)，用于梯度/速度平滑。"""
    half = (ksize - 1) / 2.0
    coords = torch.arange(ksize, device=device, dtype=dtype) - half
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2 = g.view(-1, 1) * g.view(1, -1)
    return k2.view(1, 1, ksize, ksize)


def _gaussian_smooth_2d(x: torch.Tensor, sigma: float, ksize: int = 5) -> torch.Tensor:
    """对 (B,1,H,W) 张量做高斯平滑（DiffusionFWI/ILVR 的稳定化技巧）。"""
    if sigma <= 0:
        return x
    kernel = _gaussian_kernel(sigma, ksize, x.device, x.dtype)
    pad = ksize // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel)


def _gaussian_smooth_2d_anisotropic(
    x: torch.Tensor, sigma_v: float, sigma_h: float, ksize: int = 5,
) -> torch.Tensor:
    """各向异性 2D 高斯平滑：``sigma_v``（深度/H 方向）+ ``sigma_h``（横向/W 方向）。

    对齐 ILVR 仓库 ``gaussian_filter(grad, [2, grad_smooth])`` 的语义：先沿
    H 方向 1D 卷积、再沿 W 方向 1D 卷积。任一 σ ≤ 0 退化为只在另一方向平滑。
    """
    if sigma_v <= 0 and sigma_h <= 0:
        return x
    half = (ksize - 1) / 2.0
    coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - half
    pad = ksize // 2
    out = x
    if sigma_v > 0:
        gv = torch.exp(-(coords ** 2) / (2.0 * sigma_v ** 2))
        gv = (gv / gv.sum()).view(1, 1, ksize, 1)
        out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), gv)
    if sigma_h > 0:
        gh = torch.exp(-(coords ** 2) / (2.0 * sigma_h ** 2))
        gh = (gh / gh.sum()).view(1, 1, 1, ksize)
        out = F.conv2d(F.pad(out, (pad, pad, 0, 0), mode="reflect"), gh)
    return out


def _make_snapshot_steps(n_iters: int, n_snapshots: int) -> List[int]:
    """生成 n_snapshots 个等距快照步号；含 0（初始）和 n_iters（终态）。"""
    if n_snapshots <= 0:
        return []
    if n_snapshots == 1:
        return [n_iters]
    return list(np.linspace(0, n_iters, n_snapshots, dtype=int).tolist())


def _capture_v_phys(mu_norm: torch.Tensor) -> np.ndarray:
    """归一化 v ([-1,1]) → 物理 m/s 的 numpy 数组 (H, W)。"""
    return _v_to_phys(mu_norm.detach().squeeze().float().cpu().numpy()).astype(np.float32)


def _capture_latent(z: torch.Tensor) -> np.ndarray:
    """DDIM 隐变量 z (1,1,H,W) → numpy (H, W)，保留原数值 (≈N(0,1))。"""
    return z.detach().squeeze().float().cpu().numpy().astype(np.float32)


# =============================================================================
# 物理域 FWI 公共主循环（被 Tikhonov / TV / RED-DiffEq 复用）
# =============================================================================
def _physical_fwi_loop(
    *,
    velocity_init_norm: torch.Tensor,
    velocity_true_norm: torch.Tensor,
    seismic_obs: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    reg_loss_fn: Callable[[torch.Tensor, int], Tuple[torch.Tensor, float]],
    n_iters: int,
    lr: float,
    reg_lambda: float,
    obs_loss_kind: str,
    device: torch.device,
    snapshots: int = 0,
    noise_sigma_x0: float = 0.0,
    use_scheduler: bool = False,
    trace_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]], List[Dict]]:
    """物理空间 Adam FWI；正则项由 ``reg_loss_fn`` 提供。

    ``reg_loss_fn(x0_pred_norm, step) → (reg_scalar_tensor, reg_value_for_log)``

    若 ``snapshots > 0``，在迭代过程中等距捕捉中间速度场（含初始 step=0 和
    终态 step=n_iters），返回的 ``snap_data`` = ``[{step, velocity, rmse, mae, ssim}, ...]``。

    ``noise_sigma_x0 > 0`` 时（官方 RED-DiffEq 行为）：每步对 μ 注入
    ``σ·η`` 得到 ``x0_pred``，**同时**用于 forward 物理正演和正则项，与官方
    ``red_diffeq`` 仓库 ``core/inversion.py`` 一致。Tikhonov / TV 默认 ``0``。

    ``use_scheduler=True`` 时套 ``CosineAnnealingLR(optimizer, T_max=n_iters,
    eta_min=0)``，与官方 RED-DiffEq ``InversionEngine`` 默认一致——末段 lr 衰
    减到 0 让 μ 沉淀，可显著减少高频噪点。
    """
    mu = velocity_init_norm.clone().detach().to(device).requires_grad_(True)
    target_seismic = seismic_obs.to(device)
    target_norm = velocity_true_norm.to(device)
    mask_dev = trace_mask.to(device) if trace_mask is not None else None

    optimizer = torch.optim.Adam([mu], lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=0.0)
        if use_scheduler else None
    )

    history: Dict[str, List[float]] = {
        "rmse": [], "mae": [], "ssim": [],
        "obs_loss": [], "reg_loss": [], "total_loss": [],
    }
    snap_steps = set(_make_snapshot_steps(n_iters, snapshots))
    snap_data: List[Dict] = []

    # 前置真·初始指标（高斯平滑模型 vs GT），与其它方法的 history[0] 语义对齐。
    with torch.no_grad():
        r0, m0, s0 = _metrics_norm(mu, target_norm)
    history["rmse"].append(r0); history["mae"].append(m0); history["ssim"].append(s0)
    history["obs_loss"].append(0.0); history["reg_loss"].append(0.0); history["total_loss"].append(0.0)

    if 0 in snap_steps:
        snap_data.append({"step": 0, "velocity": _capture_v_phys(mu),
                          "rmse": r0, "mae": m0, "ssim": s0})

    for step in range(n_iters):
        optimizer.zero_grad(set_to_none=True)

        if noise_sigma_x0 > 0.0:
            x0_pred = mu + torch.randn_like(mu) * noise_sigma_x0
        else:
            x0_pred = mu

        v_phys = _v_to_phys(x0_pred.squeeze(0).squeeze(0)).clamp(VELOCITY_VMIN_M_S, VELOCITY_VMAX_M_S)
        pred_seismic = forward_fn(v_phys)

        loss_obs = _obs_loss(pred_seismic, target_seismic, kind=obs_loss_kind, mask=mask_dev)

        reg_tensor, reg_log = reg_loss_fn(x0_pred, step)
        loss = loss_obs + reg_lambda * reg_tensor
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mu.data.clamp_(-1.0, 1.0)
            rmse, mae, ssim = _metrics_norm(mu, target_norm)

        if scheduler is not None:
            scheduler.step()

        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["ssim"].append(ssim)
        history["obs_loss"].append(float(loss_obs.item()))
        history["reg_loss"].append(float(reg_log))
        history["total_loss"].append(float(loss.item()))

        if (step + 1) in snap_steps:
            snap_data.append({"step": step + 1,
                              "velocity": _capture_v_phys(mu),
                              "rmse": rmse, "mae": mae, "ssim": ssim})

    return mu.detach(), history, snap_data


# =============================================================================
# 1) Tikhonov（L2 各向同性平滑）
# =============================================================================
TIKHONOV_DEFAULTS: Dict = {
    "n_iters": 300,
    "lr": 0.03,
    "reg_lambda": 0.01,
    "smooth_sigma": 10.0,
    "obs_loss": "l1",
    "use_scheduler": True,   # CosineAnnealingLR(T_max=n_iters, eta_min=0)，与 RED-DiffEq 官方 InversionEngine 一致
}


def _tikhonov_reg(mu: torch.Tensor) -> torch.Tensor:
    """:math:`\\|\\nabla\\mu\\|_2^2`，对相邻像素 L2 差。"""
    dx = mu[..., 1:] - mu[..., :-1]
    dy = mu[..., 1:, :] - mu[..., :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


def run_tikhonov(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    cfg = {**TIKHONOV_DEFAULTS, **(params or {})}

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v_init_norm = torch.from_numpy(v_init_norm_np).view(1, 1, *v_true_np.shape)
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape)

    def reg_fn(mu: torch.Tensor, step: int) -> Tuple[torch.Tensor, float]:
        r = _tikhonov_reg(mu)
        return r, float(r.item())

    mu_pred, history, snap_data = _physical_fwi_loop(
        velocity_init_norm=v_init_norm,
        velocity_true_norm=v_true_norm,
        seismic_obs=seismic_obs,
        forward_fn=forward_fn,
        reg_loss_fn=reg_fn,
        n_iters=cfg["n_iters"],
        lr=cfg["lr"],
        reg_lambda=cfg["reg_lambda"],
        obs_loss_kind=cfg["obs_loss"],
        device=device,
        snapshots=int(cfg.get("snapshots", 0)),
        use_scheduler=bool(cfg.get("use_scheduler", False)),
        trace_mask=trace_mask,
    )
    velocity_pred_phys = _v_to_phys(mu_pred.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="tikhonov",
        params=cfg,
        extra={"snapshots": snap_data},
    )


# =============================================================================
# 2) Total Variation
# =============================================================================
TV_DEFAULTS: Dict = {
    "n_iters": 300,
    "lr": 0.03,
    "reg_lambda": 0.01,
    "smooth_sigma": 10.0,
    "obs_loss": "l1",
    "use_scheduler": True,   # CosineAnnealingLR(T_max=n_iters, eta_min=0)
}


def _tv_reg(mu: torch.Tensor) -> torch.Tensor:
    """:math:`\\|\\nabla\\mu\\|_1`，各向同性近似（分别 x、y）。"""
    dx = (mu[..., 1:] - mu[..., :-1]).abs()
    dy = (mu[..., 1:, :] - mu[..., :-1, :]).abs()
    return dx.mean() + dy.mean()


def run_tv(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    cfg = {**TV_DEFAULTS, **(params or {})}

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v_init_norm = torch.from_numpy(v_init_norm_np).view(1, 1, *v_true_np.shape)
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape)

    def reg_fn(mu: torch.Tensor, step: int) -> Tuple[torch.Tensor, float]:
        r = _tv_reg(mu)
        return r, float(r.item())

    mu_pred, history, snap_data = _physical_fwi_loop(
        velocity_init_norm=v_init_norm,
        velocity_true_norm=v_true_norm,
        seismic_obs=seismic_obs,
        forward_fn=forward_fn,
        reg_loss_fn=reg_fn,
        n_iters=cfg["n_iters"],
        lr=cfg["lr"],
        reg_lambda=cfg["reg_lambda"],
        obs_loss_kind=cfg["obs_loss"],
        device=device,
        snapshots=int(cfg.get("snapshots", 0)),
        use_scheduler=bool(cfg.get("use_scheduler", False)),
        trace_mask=trace_mask,
    )
    velocity_pred_phys = _v_to_phys(mu_pred.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="tv",
        params=cfg,
        extra={"snapshots": snap_data},
    )


# =============================================================================
# 3) RED-DiffEq (Shan et al. 2026)
# =============================================================================
RED_DIFFEQ_DEFAULTS: Dict = {
    "n_iters": 300,
    "lr": 0.03,
    "reg_lambda": 0.75,                 # SI Sec.5: 0.75 for OpenFWI
    "smooth_sigma": 10.0,
    "sigma_x0": 1e-4,                   # μ → μ + σ·η before q_sample
    "fixed_timestep": None,             # None ⇒ uniform[0, T)
    "use_time_weight": False,           # SI Sec.6: 固定权重在 OpenFWI 上更好
    "obs_loss": "l1",
    "use_scheduler": True,              # 官方 InversionEngine 写死 CosineAnnealingLR(T_max=ts, eta_min=0)
}


def _red_diffeq_grad_field(
    x0_pred: torch.Tensor,
    diffusion: DiffusionPrior,
    fixed_timestep: Optional[int],
    use_time_weight: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RED-DiffEq 梯度场 :math:`\\epsilon_\\theta(x_t,t) - \\epsilon`，detach 后乘 x0_pred。

    ``x0_pred`` 已在外层主循环中由 μ + σ·η 构造（与官方 ``red_diffeq``
    ``core/inversion.py`` 行为一致）。返回 (reg_loss_for_backward, time_t)。
    """
    B = x0_pred.shape[0]
    device = x0_pred.device
    T_max = diffusion.num_train_timesteps if fixed_timestep is None else int(fixed_timestep)
    t = torch.randint(0, T_max, (B,), device=device, dtype=torch.long)

    noise = torch.randn_like(x0_pred)
    abar = diffusion.alphas_cumprod[t].to(device).float().view(B, 1, 1, 1)
    x_t = abar.sqrt() * x0_pred + (1.0 - abar).sqrt() * noise

    eps_raw = diffusion.denoise(x_t, t)

    pred_x0 = ((x_t - (1.0 - abar).sqrt() * eps_raw) / abar.sqrt()).clamp(-1.0, 1.0)
    eps_pred = (x_t - abar.sqrt() * pred_x0) / (1.0 - abar).sqrt()

    grad_field = (eps_pred - noise).detach()

    if use_time_weight:
        w_t = ((1.0 - abar) / abar).sqrt()
        grad_field = grad_field * w_t

    reg_field = grad_field * x0_pred
    reg_loss = reg_field.mean()
    return reg_loss, t


def run_red_diffeq(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    if diffusion is None:
        raise ValueError("RED-DiffEq requires a DiffusionPrior.")
    cfg = {**RED_DIFFEQ_DEFAULTS, **(params or {})}

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v_init_norm = torch.from_numpy(v_init_norm_np).view(1, 1, *v_true_np.shape)
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape)

    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)

    def reg_fn(x0_pred: torch.Tensor, step: int) -> Tuple[torch.Tensor, float]:
        reg_loss, _t = _red_diffeq_grad_field(
            x0_pred, diffusion,
            fixed_timestep=cfg["fixed_timestep"],
            use_time_weight=cfg["use_time_weight"],
        )
        return reg_loss, float(reg_loss.item())

    mu_pred, history, snap_data = _physical_fwi_loop(
        velocity_init_norm=v_init_norm,
        velocity_true_norm=v_true_norm,
        seismic_obs=seismic_obs,
        forward_fn=forward_fn,
        reg_loss_fn=reg_fn,
        n_iters=cfg["n_iters"],
        lr=cfg["lr"],
        reg_lambda=cfg["reg_lambda"],
        obs_loss_kind=cfg["obs_loss"],
        device=device,
        snapshots=int(cfg.get("snapshots", 0)),
        noise_sigma_x0=float(cfg["sigma_x0"]),
        use_scheduler=bool(cfg.get("use_scheduler", False)),
        trace_mask=trace_mask,
    )
    velocity_pred_phys = _v_to_phys(mu_pred.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="red_diffeq",
        params=cfg,
        extra={"snapshots": snap_data},
    )


# =============================================================================
# 4) DiffusionFWI (Wang et al. 2023)
# =============================================================================
DIFFUSION_FWI_DEFAULTS: Dict = {
    # ⚠ 语义与 Taufik ilvrefwi/diffefwi ``fwi_sample`` 同名参数一致：
    #   "从 t=T−1 开始跳过的高噪步数"，反向步数 = num_train_timesteps − init_time_step。
    #   例：num_train=1000, init_time_step=900 ⇒ 从 t=99 倒着走到 0，共 100 反向步。
    "init_time_step": 900,
    "fwi_iters_per_step": 10,      # 每个扩散步内的 FWI 梯度步数（官方 args.fwi_iteration）
    "lr": 0.03,
    "smooth_sigma": 10.0,          # 初始模型高斯平滑 σ
    "obs_loss": "l1",
    # ── 优化器/调度器（对齐官方 fwi_loop 默认）──
    "optim": "adam",               # 'adam' | 'lbfgs'（官方亦支持 LBFGS）
    "use_scheduler": False,        # True ⇒ CosineAnnealingLR(optimizer, K+1, 0)（官方可选）
    "grad_clip": 1.0,              # 官方 fwi_loop 默认 grad_clip=1.0；None=关闭
    # ── 稳定化（对齐 Taufik ilvrefwi 仓库 fwi.py:fwi_loop 三件套）──
    # ① 梯度平滑：ILVR ``gaussian_filter(grad, [2, gaussian_window])``；
    #    sigma_v=2.0 是官方写死的深度方向 σ；sigma=1.0 对应 Example-2-efwi.py
    #    的 ``grad_smooth=1`` 横向 σ。sigma_v=0 时退化为各向同性。
    "grad_smooth_sigma": 1.0,
    "grad_smooth_sigma_v": 2.0,    # 官方写死 [2, sigma_h]
    "grad_smooth_kernel": 5,
    # ② 速度模型模糊（每个 FWI inner step 开头做）：ILVR 写死 ``gaussian_blur(v, [3,3])``
    "velocity_blur_kernel": 3,     # 0 表示关闭；3 = ILVR 默认
    "velocity_blur_sigma": 0.8,    # torchvision gaussian_blur 默认 σ(kernel=3) ≈ 0.8
    # ③ 梯度归一化（ILVR step-0 baseline）：官方 Example-2-efwi.py grad_norm=True
    "grad_normalize": True,
    # 旧接口（per-diffusion-step 速度模糊，每个 t 末尾做一次）：默认关
    "vel_blur_sigma": 0.0,
    "vel_blur_kernel": 3,
    # 反向步噪声缩放：对齐 Taufik ilvrefwi/diffefwi ``p_sample_wf`` 写死的
    # ``noise * 0.001``，即"DDPM μ_t + 微弱噪声"近确定性采样。1.0 = 标准
    # DDPM 祖先采样（fixed_small）；0.0 = 完全确定性（仅 μ_t）。
    "ddpm_noise_scale": 0.001,
}


def _fwi_inner_loop(
    v_norm_init: torch.Tensor,
    *,
    target_seismic: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int,
    lr: float,
    obs_loss_kind: str,
    grad_smooth_sigma: float,
    grad_smooth_kernel: int,
    grad_smooth_sigma_v: float = 0.0,
    velocity_blur_kernel: int = 0,
    velocity_blur_sigma: float = 0.8,
    grad_normalize: bool = False,
    optim: str = "adam",
    use_scheduler: bool = False,
    grad_clip: Optional[float] = 1.0,
    trace_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """K 步纯 FWI（[-1,1] 域），返回 (v_norm_final, obs_loss_history)。

    与 Taufik ilvrefwi 仓库 ``src/ilvrefwi/fwi.py::fwi_loop`` 1:1 对齐：
      操作顺序（每 step）：
        ① 边界投影 v.clamp_(-1,1) + 速度模糊 gaussian_blur(v,[3,3])
        ② zero_grad → forward → backward
        ③ step==0 记录 grad_baseline = |grad|_∞
        ④ grad_normalize：v.grad /= grad_baseline
        ⑤ grad_smooth：高斯平滑 grad，**并刷新** grad_baseline = |smoothed grad|_∞
        ⑥ grad_clip：clip_grad_norm_(v, grad_clip × grad_baseline)
        ⑦ optimizer.step() ; scheduler.step()（若开启）

    参数：
      optim:           'adam' | 'lbfgs'（官方默认 adam）。
      use_scheduler:   True ⇒ CosineAnnealingLR(optimizer, n_steps+1, 0)。
      grad_clip:       L2-norm 上界系数；max_norm = grad_clip × grad_baseline。
                       None 关闭；官方默认 1.0。
      grad_normalize:  True ⇒ 用 step-0 |grad|_∞ 把梯度归一化为 ≈1 量级。
      grad_smooth_*:   横向 σ + 深度 σ；σ_v=0 退化各向同性。
    """
    v = v_norm_init.detach().clone().requires_grad_(True)
    if optim == "adam":
        optimizer = torch.optim.Adam([{"params": [v], "lr": lr}])
    elif optim == "lbfgs":
        optimizer = torch.optim.LBFGS([v])
    else:
        raise ValueError(f"unknown optim: {optim!r}")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps + 1, 0)
        if use_scheduler else None
    )

    losses: List[float] = []
    grad_baseline: Optional[float] = None   # 持久变量：step-0 设置，可能在 grad_smooth 块内刷新

    for step in range(n_steps):
        # ① Bounds projection + velocity blur （官方 fwi.py:163-183，先 clamp 后 blur，无后置 clamp）
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)
            if velocity_blur_kernel > 0:
                v.data = _gaussian_smooth_2d(v.data, velocity_blur_sigma, velocity_blur_kernel)

        # ② forward / backward
        optimizer.zero_grad()
        v_phys = _v_to_phys(v.squeeze(0).squeeze(0)).clamp(VELOCITY_VMIN_M_S, VELOCITY_VMAX_M_S)
        pred = forward_fn(v_phys)
        loss = _obs_loss(pred, target_seismic, kind=obs_loss_kind, mask=trace_mask)
        loss.backward()

        if v.grad is not None:
            # ③ step-0 baseline （官方 fwi.py:323-326，从 raw grad 取）
            if step == 0:
                grad_baseline = float(v.grad.abs().max().item())

            # ④ grad_normalize （官方 fwi.py:328-335，除以 baseline）
            if grad_normalize and grad_baseline is not None and grad_baseline > 0:
                v.grad.data.div_(grad_baseline)

            # ⑤ grad_smooth + 刷新 baseline （官方 fwi.py:350-357）
            if grad_smooth_sigma > 0:
                if grad_smooth_sigma_v > 0:
                    v.grad.data = _gaussian_smooth_2d_anisotropic(
                        v.grad.data, grad_smooth_sigma_v, grad_smooth_sigma, grad_smooth_kernel,
                    )
                else:
                    v.grad.data = _gaussian_smooth_2d(
                        v.grad.data, grad_smooth_sigma, grad_smooth_kernel,
                    )
                grad_baseline = float(v.grad.abs().max().item())

            # ⑥ grad_clip （官方 fwi.py:359-362，clip_grad_norm_ 到 grad_clip * baseline）
            if grad_clip is not None and grad_baseline is not None and grad_baseline > 0:
                torch.nn.utils.clip_grad_norm_([v], max_norm=grad_clip * grad_baseline)

        # ⑦ optimizer step
        if optim == "lbfgs":
            optimizer.step(lambda: loss)   # 官方 fwi_loop 也是无 closure 调用；这里给个最小 closure 兼容新版 PyTorch
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
    return v.detach(), losses


def run_diffusion_fwi(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    """DiffusionFWI（Wang et al. 2023）— 与 Taufik ilvrefwi/diffefwi 对齐。

    主循环结构：
      - 从平滑初始模型 v₀ 开始；
      - 反向时间步 t = T−init_time_step−1, …, 0（共 T−init_time_step 步，
        ``init_time_step`` 即官方语义"跳过的高噪步数"）：
          1. 视当前 v 为 x_t：ε̂ = ε_θ(v, t), x̂_0 = clip((v-√(1-ᾱ)ε̂)/√ᾱ);
          2. 取 DDPM 反向后验均值 :math:`\\mu(\\hat x_0, v, t)` + 弱噪声
             ``ddpm_noise_scale · σ_t · z`` (默认 0.001，近确定性) 作为 v_seed；
          3. 在 v_seed 上跑 K 步 FWI 梯度下降（带梯度高斯平滑）；
          4. 将结果写回 v；
      - 末步 t=0 输出 v 即反演结果。
    """
    if diffusion is None:
        raise ValueError("DiffusionFWI requires a DiffusionPrior.")
    cfg = {**DIFFUSION_FWI_DEFAULTS, **(params or {})}

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v = torch.from_numpy(v_init_norm_np).view(1, 1, *v_true_np.shape).to(device).clamp(-1.0, 1.0)
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape).to(device)
    target_seismic = seismic_obs.to(device)
    mask_dev = trace_mask.to(device) if trace_mask is not None else None

    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)

    init_t = int(cfg["init_time_step"])
    K = int(cfg["fwi_iters_per_step"])
    noise_scale = float(cfg.get("ddpm_noise_scale", 0.001))
    n_reverse_steps = diffusion.num_train_timesteps - init_t

    history: Dict[str, List[float]] = {
        "rmse": [], "mae": [], "ssim": [],
        "obs_loss": [], "reg_loss": [], "total_loss": [],
    }

    with torch.no_grad():
        rmse, mae, ssim = _metrics_norm(v, v_true_norm)
    history["rmse"].append(rmse); history["mae"].append(mae); history["ssim"].append(ssim)
    history["obs_loss"].append(0.0); history["reg_loss"].append(0.0); history["total_loss"].append(0.0)

    for t_curr in reversed(range(n_reverse_steps)):
        # ---- diffusion 一步：DDPM 反向 x_t → x_{t-1}（μ + ddpm_noise_scale·σ·z）----
        # 把当前 v 当 x_t，与 Taufik ilvrefwi/diffefwi ``p_sample_wf`` 等价：
        #   ε̂ = ε_θ(v, t) → x̂₀(clip) → posterior μ → +0.001·σ_t·z（默认）
        with torch.no_grad():
            t_tensor = torch.tensor([t_curr], device=device, dtype=torch.long)
            eps_pred = diffusion.denoise(v, t_tensor)
            v_seed, _ = _ddpm_step(
                v, eps_pred, t_curr, diffusion.alphas_cumprod,
                clip_x0=True, noise_scale=noise_scale,
            )
            v_seed = v_seed.clamp(-1.0, 1.0)

        # ---- K 步 FWI 在去噪后的速度上做梯度下降 ----
        v, fwi_losses = _fwi_inner_loop(
            v_seed,
            target_seismic=target_seismic,
            forward_fn=forward_fn,
            n_steps=K,
            lr=cfg["lr"],
            obs_loss_kind=cfg["obs_loss"],
            grad_smooth_sigma=cfg["grad_smooth_sigma"],
            grad_smooth_sigma_v=cfg.get("grad_smooth_sigma_v", 0.0),
            grad_smooth_kernel=cfg["grad_smooth_kernel"],
            velocity_blur_kernel=int(cfg.get("velocity_blur_kernel", 0)),
            velocity_blur_sigma=float(cfg.get("velocity_blur_sigma", 0.8)),
            grad_normalize=bool(cfg.get("grad_normalize", False)),
            optim=str(cfg.get("optim", "adam")),
            use_scheduler=bool(cfg.get("use_scheduler", False)),
            grad_clip=cfg.get("grad_clip", 1.0),
            trace_mask=mask_dev,
        )

        # 官方 fwi_sample 在每个扩散步末尾做无条件 clamp（diffusion.py:683-685）
        v = v.clamp(-1.0, 1.0)
        # 旧接口：per-diffusion-step 速度域模糊（默认关）
        if cfg["vel_blur_sigma"] > 0:
            v = _gaussian_smooth_2d(v, cfg["vel_blur_sigma"], cfg["vel_blur_kernel"])
            v = v.clamp(-1.0, 1.0)

        with torch.no_grad():
            rmse, mae, ssim = _metrics_norm(v, v_true_norm)
        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["ssim"].append(ssim)
        last_obs = fwi_losses[-1] if fwi_losses else 0.0
        history["obs_loss"].append(last_obs)
        history["reg_loss"].append(0.0)
        history["total_loss"].append(last_obs)

    velocity_pred_phys = _v_to_phys(v.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="diffusion_fwi",
        params=cfg,
        extra={"total_inner_fwi_steps": n_reverse_steps * K},
    )


# =============================================================================
# 5) DiffusionILVR
# =============================================================================
DIFFUSION_ILVR_DEFAULTS: Dict = {
    **DIFFUSION_FWI_DEFAULTS,
    # 默认对齐 Taufik (ilvrefwi) 仓库的实现风格
    "ilvr_factor": 4,                  # 下采样倍数 N（factor_schedule 为空时使用）
    "ilvr_factor_schedule": None,      # 例：[32,16,8,4]；非空时按 t_curr 索引取 factor
                                       # （等距重复到 init_time_step 长度），与官方 down_n 行为一致
    "ilvr_weight": 0.05,        # ilvrefwi default：弱低频引导
    "ilvr_range_t": 0,          # 仅当 t > range_t 时启用；0 = 全程开启
    "ilvr_ref": "current",      # 'current' (Taufik) | 'init' (经典 fixed-ref) | 'gt_low' (oracle)
    "ilvr_domain": "xt",        # 'xt' (Taufik，q_sample 当前 v 到 t) | 'x0' (直接用 ref_x0)
}


def _expand_ilvr_schedule(schedule: Optional[List[int]], n_steps: int, fallback: int) -> List[int]:
    """把短 schedule 等距铺到 n_steps 长度；空/None 时返回 [fallback]*n_steps。

    与官方 ``np.repeat(Ns, n_steps // len(Ns))`` 等价；不能整除时末尾补最后一档。
    """
    if not schedule:
        return [fallback] * n_steps
    k = len(schedule)
    seg = n_steps // k
    out: List[int] = []
    for f in schedule:
        out.extend([int(f)] * seg)
    while len(out) < n_steps:
        out.append(int(schedule[-1]))
    return out[:n_steps]


def _ilvr_low_pass_replace(
    x: torch.Tensor,
    ref: torch.Tensor,
    factor: int,
    weight: float,
) -> torch.Tensor:
    """ILVR 低频替换：``x ← x − w·LP(x) + w·LP(ref)``。

    要求 x 与 ref 处在 *同一噪声层* —— 否则会因量纲差异引入大量噪声。
    """
    if factor <= 1 or weight <= 0:
        return x
    lp_x = _low_pass_avgpool(x, factor)
    lp_ref = _low_pass_avgpool(ref, factor)
    return x - weight * lp_x + weight * lp_ref


def run_diffusion_ilvr(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    """DiffusionFWI + ILVR：每个反向扩散步先用低频对齐参考再做 K 步 FWI。

    参考图 ``ilvr_ref``：
      - ``"current"`` (默认, Taufik/ilvrefwi)  → 用当前 v（去噪步开始前的状态）作 ref；
                              结合 ``ilvr_domain="xt"`` 即 ``LP(q_sample(v, t, ε))``，
                              语义上"保留当前 v 的低频，去噪只在高频起作用"；
      - ``"init"``    → 固定参考 = 平滑初始速度 v_init（经典 ILVR 风格）；
      - ``"gt_low"``  → 低通 ground truth（仅作 oracle/上限调试，不用于实测）。

    替换域 ``ilvr_domain``：
      - ``"xt"`` (默认)  → ref 先 q_sample(·, t, ε) 加噪到 t 噪声层再 LP；
      - ``"x0"``         → 直接用 LP(ref_x0)，不加噪（量纲与 v_seed≈x_0 域匹配）。
    """
    if diffusion is None:
        raise ValueError("DiffusionILVR requires a DiffusionPrior.")
    cfg = {**DIFFUSION_ILVR_DEFAULTS, **(params or {})}

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v = torch.from_numpy(v_init_norm_np).view(1, 1, *v_true_np.shape).to(device).clamp(-1.0, 1.0)
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape).to(device)
    target_seismic = seismic_obs.to(device)
    mask_dev = trace_mask.to(device) if trace_mask is not None else None

    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)

    init_t = int(cfg["init_time_step"])
    K = int(cfg["fwi_iters_per_step"])
    noise_scale = float(cfg.get("ddpm_noise_scale", 0.001))
    n_reverse_steps = diffusion.num_train_timesteps - init_t
    factor_schedule = _expand_ilvr_schedule(
        cfg.get("ilvr_factor_schedule"), n_reverse_steps, int(cfg["ilvr_factor"])
    )
    weight = float(cfg["ilvr_weight"])
    range_t = int(cfg["ilvr_range_t"])
    ilvr_domain = str(cfg.get("ilvr_domain", "xt"))
    ilvr_ref_mode = str(cfg.get("ilvr_ref", "current"))

    # 固定参考一次性获取（仅 ref ∈ {init, gt_low} 用，'current' 模式每步都从 v 取）
    if ilvr_ref_mode == "init":
        fixed_ref = v.clone()
    elif ilvr_ref_mode == "gt_low":
        fixed_ref = v_true_norm.clone()
    elif ilvr_ref_mode == "current":
        fixed_ref = None
    else:
        raise ValueError(f"unknown ilvr_ref: {ilvr_ref_mode}")

    history: Dict[str, List[float]] = {
        "rmse": [], "mae": [], "ssim": [],
        "obs_loss": [], "reg_loss": [], "total_loss": [],
    }
    with torch.no_grad():
        rmse, mae, ssim = _metrics_norm(v, v_true_norm)
    history["rmse"].append(rmse); history["mae"].append(mae); history["ssim"].append(ssim)
    history["obs_loss"].append(0.0); history["reg_loss"].append(0.0); history["total_loss"].append(0.0)

    for t_curr in reversed(range(n_reverse_steps)):
        with torch.no_grad():
            # ---- DDPM 反向一步 x_t → x_{t-1} （与 run_diffusion_fwi 一致） ----
            t_tensor = torch.tensor([t_curr], device=device, dtype=torch.long)
            eps_pred = diffusion.denoise(v, t_tensor)
            v_seed, _ = _ddpm_step(
                v, eps_pred, t_curr, diffusion.alphas_cumprod,
                clip_x0=True, noise_scale=noise_scale,
            )

            # ---- ILVR：低频替换（叠加在 DDPM 反向结果上） ----
            ref_x0 = v if ilvr_ref_mode == "current" else fixed_ref
            factor = factor_schedule[t_curr]
            if t_curr > range_t and factor > 1:
                if ilvr_domain == "xt":
                    # Taufik/ilvrefwi 实现：把 ref_x0 加噪到 t 噪声层再取低频
                    ref_noise = torch.randn_like(ref_x0)
                    ref_used = _q_sample(ref_x0, t_curr, diffusion.alphas_cumprod, ref_noise)
                elif ilvr_domain == "x0":
                    # 在 x_0 域直接用 ref_x0（量纲与 v_seed≈x_0 匹配，不引入 q_sample 噪声）
                    ref_used = ref_x0
                else:
                    raise ValueError(f"unknown ilvr_domain: {ilvr_domain}")
                v_seed = _ilvr_low_pass_replace(v_seed, ref_used, factor, weight)

            v_seed = v_seed.clamp(-1.0, 1.0)

        v, fwi_losses = _fwi_inner_loop(
            v_seed,
            target_seismic=target_seismic,
            forward_fn=forward_fn,
            n_steps=K,
            lr=cfg["lr"],
            obs_loss_kind=cfg["obs_loss"],
            grad_smooth_sigma=cfg["grad_smooth_sigma"],
            grad_smooth_sigma_v=cfg.get("grad_smooth_sigma_v", 0.0),
            grad_smooth_kernel=cfg["grad_smooth_kernel"],
            velocity_blur_kernel=int(cfg.get("velocity_blur_kernel", 0)),
            velocity_blur_sigma=float(cfg.get("velocity_blur_sigma", 0.8)),
            grad_normalize=bool(cfg.get("grad_normalize", False)),
            optim=str(cfg.get("optim", "adam")),
            use_scheduler=bool(cfg.get("use_scheduler", False)),
            grad_clip=cfg.get("grad_clip", 1.0),
            trace_mask=mask_dev,
        )

        # 官方 fwi_sample 在每个扩散步末尾做无条件 clamp
        v = v.clamp(-1.0, 1.0)
        if cfg["vel_blur_sigma"] > 0:
            v = _gaussian_smooth_2d(v, cfg["vel_blur_sigma"], cfg["vel_blur_kernel"])
            v = v.clamp(-1.0, 1.0)

        with torch.no_grad():
            rmse, mae, ssim = _metrics_norm(v, v_true_norm)
        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["ssim"].append(ssim)
        last_obs = fwi_losses[-1] if fwi_losses else 0.0
        history["obs_loss"].append(last_obs)
        history["reg_loss"].append(0.0)
        history["total_loss"].append(last_obs)

    velocity_pred_phys = _v_to_phys(v.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="diffusion_ilvr",
        params=cfg,
        extra={"total_inner_fwi_steps": n_reverse_steps * K},
    )


# =============================================================================
# 6) DLO-FWI — Decoupled Latent Optimization for FWI（本工作）
#    解耦的双变量潜空间优化：物理速度 v 与扩散潜变量 z 在两条独立轨迹上交替更新，
#    通过 L2 软耦合相互引导。区别于 DMPlug / D-Flow 等单变量 z 上的 latent
#    optimization，DLO-FWI 把"物理梯度稳定性"与"流形约束"解耦到不同变量。
# =============================================================================
DLO_FWI_DEFAULTS: Dict = {
    "n_iters": 300,
    "lr_v": 0.03,
    "lr_z": 0.03,
    "z_steps_per_iter": 1,
    "smooth_sigma": 10.0,
    "obs_loss": "l1",
    "lambda_max": 0.75,
    "warmup_steps": 50,
    "ramp_steps": 100,
    "ddim_steps": 6,
    "ddim_eta": 0.0,
    # 只对 phase1 的 opt_v 套 cosine（opt_z 保持恒定，z 的更新本来就是辅助变量）
    "use_scheduler": True,
    # DDIM 步内开关（DMPlug 风格，默认开启）：clip pred_x0 + rederive ε
    "ddim_clip_sample": True,
    "ddim_clip_sample_range": 1.0,
    "ddim_use_clipped_model_output": True,
    "phase2_steps": 100,           # 0 ⇒ 关闭 phase2
    "phase2_lr": 5e-4,
    "phase2_ddim_steps": 3,
}


def _build_ddim_timesteps_full(num_steps: int, num_train: int) -> List[int]:
    """从 t=num_train-1 等距递减到 0（与 diffusers DDIMScheduler set_timesteps 一致）。"""
    if num_steps <= 1:
        return [num_train - 1]
    step = num_train / num_steps
    return [int(round((num_steps - 1 - i) * step)) for i in range(num_steps)]


def _ddim_sample(
    z: torch.Tensor,
    diffusion: DiffusionPrior,
    num_steps: int,
    eta: float = 0.0,
    require_grad: bool = True,
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    use_clipped_model_output: bool = True,
) -> torch.Tensor:
    """DDIM 反向采样 z(噪声) → x_0；require_grad=False 时禁用梯度。

    与 diffusers ``DDIMScheduler.step`` 行为等价。两个关键开关（DMPlug 风格，
    默认开启）：

    * ``clip_sample`` (default ``True``)：每步对预测的 ``x_0`` clamp 到
      ``[-clip_sample_range, +clip_sample_range]``，避免越界累积。
    * ``use_clipped_model_output`` (default ``True``)：用 clipped ``x_0``
      反推 ``ε``，让 "direction pointing to x_t" 与 clipped ``x_0`` 自洽 ——
      在用 DDIM 链做潜变量优化（梯度反传穿过整个链）时尤其重要，提升稳定性
      与细节恢复。
    """
    timesteps = _build_ddim_timesteps_full(num_steps, diffusion.num_train_timesteps)
    abar = diffusion.alphas_cumprod.to(z.device).float()

    def _step():
        x = z
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            eps = diffusion.denoise(x, int(t))
            abar_t = abar[int(t)]
            abar_prev = abar[int(t_prev)] if t_prev >= 0 else torch.tensor(diffusion.final_alpha_cumprod, device=z.device, dtype=torch.float32)

            # (3) 从预测噪声解出 x̂_0
            pred_x0 = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt()
            # (4) 可选 clip x̂_0
            if clip_sample:
                pred_x0 = pred_x0.clamp(-clip_sample_range, clip_sample_range)

            # (5) 方差 / sigma
            variance = ((1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev)).clamp(min=0.0)
            sigma = eta * variance.sqrt()

            # (5') 可选：用 clipped x̂_0 反推 ε，使 direction 与 x̂_0 自洽
            eps_used = eps
            if use_clipped_model_output:
                eps_used = (x - abar_t.sqrt() * pred_x0) / (1.0 - abar_t).sqrt()

            # (6) direction pointing to x_t
            coef_dir = (1.0 - abar_prev - sigma ** 2).clamp(min=0.0).sqrt()
            # (7) x_{t-1}
            x = abar_prev.sqrt() * pred_x0 + coef_dir * eps_used
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    if require_grad:
        return _step()
    with torch.no_grad():
        return _step()


def run_dlo_fwi(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    """**DLO-FWI** — Decoupled Latent Optimization for FWI.

    解耦的双变量潜空间优化（Decoupled CSGM-style：把单一 z-only 优化
    解耦为物理 v 与潜变量 z 两条独立轨迹，二者通过 L2 软耦合相互引导）：

        每个外层迭代：
          (1) Update z, ``z_steps_per_iter`` 步：min ‖DDIM(z) − v.detach()‖²
          (2) v_gen = DDIM(z)            (no_grad)
          (3) Update v, 1 步：L_wave(v) + λ(i)·‖v − v_gen.detach()‖²

        λ warmup：前 ``warmup_steps`` 步 λ=0，之后线性 ramp 至 ``lambda_max``.

        Phase 2 (可选)：再用 wave loss 直接微调 z (``phase2_steps`` 步)。
    """
    if diffusion is None:
        raise ValueError("DLO-FWI requires a DiffusionPrior.")
    cfg = {**DLO_FWI_DEFAULTS, **(params or {})}

    # DDIM 共用 kwargs（clip + rederive 等）
    ddim_kwargs = dict(
        clip_sample=bool(cfg.get("ddim_clip_sample", True)),
        clip_sample_range=float(cfg.get("ddim_clip_sample_range", 1.0)),
        use_clipped_model_output=bool(cfg.get("ddim_use_clipped_model_output", True)),
    )

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape).to(device)
    target_seismic = seismic_obs.to(device)
    mask_dev = trace_mask.to(device) if trace_mask is not None else None

    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)

    H, W = v_true_np.shape
    v = torch.from_numpy(v_init_norm_np).view(1, 1, H, W).to(device).clamp(-1.0, 1.0).requires_grad_(True)
    z = torch.randn(1, 1, H, W, device=device, dtype=torch.float32, requires_grad=True)

    n_iters = int(cfg["n_iters"])
    opt_v = torch.optim.Adam([v], lr=cfg["lr_v"])
    opt_z = torch.optim.Adam([z], lr=cfg["lr_z"])
    # 仅给物理速度 v 的 Adam 套 cosine（与官方 RED-DiffEq 一致）；z 是辅助变量，恒定 lr。
    sched_v = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=n_iters, eta_min=0.0)
        if bool(cfg.get("use_scheduler", False)) else None
    )

    def _lambda(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return 0.0
        progress = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["ramp_steps"]))
        return cfg["lambda_max"] * progress

    # 约定：history["rmse"] 跟踪 velocity_pred 对应的变量
    #   - phase2 关闭：velocity_pred = v   ⇒ history["rmse"] = v 的指标
    #   - phase2 开启：velocity_pred = DDIM(z2)；phase1 段仍记 v，phase2 段记 v_final
    # history["rmse_vgen"] 始终跟踪 v_gen = DDIM(z) 的指标（phase1 期间生成）
    history: Dict[str, List[float]] = {
        "rmse": [], "mae": [], "ssim": [],
        "rmse_vgen": [], "mae_vgen": [], "ssim_vgen": [],
        "obs_loss": [], "reg_loss": [], "total_loss": [],
    }
    z_hist: List[float] = []
    lam_hist: List[float] = []

    # ── 快照配置：phase1 / phase2 各自取等距快照 ─────────────────────────────
    n_snap = int(cfg.get("snapshots", 0))
    phase2_iters_cfg = int(cfg.get("phase2_steps", 0))
    snap_p1_set = set(_make_snapshot_steps(n_iters, n_snap))
    snap_p2_set = set(_make_snapshot_steps(phase2_iters_cfg, n_snap)) if phase2_iters_cfg > 0 else set()
    snap_data: List[Dict] = []

    # 前置真·初始指标：history["rmse"] = 平滑初始 v 的指标，history["rmse_vgen"] = 随机 z DDIM 解码后的指标
    with torch.no_grad():
        v_gen_init = _ddim_sample(z, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                                   require_grad=False, **ddim_kwargs)
        r0, m0, s0 = _metrics_norm(v, v_true_norm)
        r0g, m0g, s0g = _metrics_norm(v_gen_init, v_true_norm)
    history["rmse"].append(r0); history["mae"].append(m0); history["ssim"].append(s0)
    history["rmse_vgen"].append(r0g); history["mae_vgen"].append(m0g); history["ssim_vgen"].append(s0g)
    history["obs_loss"].append(0.0); history["reg_loss"].append(0.0); history["total_loss"].append(0.0)

    if 0 in snap_p1_set:
        snap_data.append({
            "phase": "phase1", "step": 0,
            "velocity_v": _capture_v_phys(v),
            "velocity_vgen": _capture_v_phys(v_gen_init),
            "latent_z": _capture_latent(z),
            "rmse": r0, "mae": m0, "ssim": s0,
            "rmse_vgen": r0g, "mae_vgen": m0g, "ssim_vgen": s0g,
        })

    for step in range(n_iters):
        # ── Step 1: update z ────────────────────────────────────────────────
        loss_z_val = 0.0
        for _ in range(int(cfg["z_steps_per_iter"])):
            opt_z.zero_grad(set_to_none=True)
            v_gen_z = _ddim_sample(z, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                                   require_grad=True, **ddim_kwargs)
            loss_z = F.mse_loss(v_gen_z, v.detach())
            loss_z.backward()
            opt_z.step()
            loss_z_val = float(loss_z.item())

        # ── Step 2: decode v_gen ─────────────────────────────────────────────
        v_gen = _ddim_sample(z, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                             require_grad=False, **ddim_kwargs)

        # ── Step 3: update v ─────────────────────────────────────────────────
        lam = _lambda(step)
        opt_v.zero_grad(set_to_none=True)
        v_phys = _v_to_phys(v.squeeze(0).squeeze(0)).clamp(VELOCITY_VMIN_M_S, VELOCITY_VMAX_M_S)
        loss_obs = _obs_loss(forward_fn(v_phys), target_seismic, kind=cfg["obs_loss"], mask=mask_dev)
        if lam > 0:
            loss_guide = F.mse_loss(v, v_gen.detach())
            loss = loss_obs + lam * loss_guide
            reg_log = float(loss_guide.item())
        else:
            loss = loss_obs
            reg_log = 0.0
        loss.backward()
        opt_v.step()
        if sched_v is not None:
            sched_v.step()
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        # history["rmse"] 跟踪物理 v（与 phase2_steps=0 时的 velocity_pred 对齐）；
        # v_gen = DDIM(z) 的指标存到 "rmse_vgen"。
        with torch.no_grad():
            rmse, mae, ssim = _metrics_norm(v, v_true_norm)
            rmse_vgen, mae_vgen, ssim_vgen = _metrics_norm(v_gen, v_true_norm)
        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["ssim"].append(ssim)
        history["rmse_vgen"].append(rmse_vgen)
        history["mae_vgen"].append(mae_vgen)
        history["ssim_vgen"].append(ssim_vgen)
        history["obs_loss"].append(float(loss_obs.item()))
        history["reg_loss"].append(reg_log)
        history["total_loss"].append(float(loss.item()))
        z_hist.append(loss_z_val)
        lam_hist.append(lam)

        if (step + 1) in snap_p1_set:
            with torch.no_grad():
                rg, mg, sg = _metrics_norm(v_gen, v_true_norm)
            snap_data.append({
                "phase": "phase1", "step": step + 1,
                "velocity_v": _capture_v_phys(v),
                "velocity_vgen": _capture_v_phys(v_gen),
                "latent_z": _capture_latent(z),
                "rmse": rmse, "mae": mae, "ssim": ssim,
                "rmse_vgen": rg, "mae_vgen": mg, "ssim_vgen": sg,
            })

    # ── Phase 2 (optional): refine z directly via wave loss ──────────────────
    phase2_iters = int(cfg.get("phase2_steps", 0))
    phase2_used = False
    if phase2_iters > 0:
        phase2_used = True
        z2 = z.detach().clone().requires_grad_(True)
        opt2 = torch.optim.Adam([z2], lr=cfg["phase2_lr"])

        # phase2 起点的快照（来自 phase1 末态对应的 DDIM 解码）
        if 0 in snap_p2_set:
            with torch.no_grad():
                v2_init = _ddim_sample(z2, diffusion, cfg["phase2_ddim_steps"],
                                       cfg["ddim_eta"], require_grad=False,
                                       **ddim_kwargs).clamp(-1.0, 1.0)
                r2, m2, s2 = _metrics_norm(v2_init, v_true_norm)
            snap_data.append({
                "phase": "phase2", "step": 0,
                "velocity_v": _capture_v_phys(v2_init),
                "velocity_vgen": _capture_v_phys(v2_init),
                "latent_z": _capture_latent(z2),
                "rmse": r2, "mae": m2, "ssim": s2,
                "rmse_vgen": r2, "mae_vgen": m2, "ssim_vgen": s2,
            })

        for p2_step in range(phase2_iters):
            opt2.zero_grad(set_to_none=True)
            v2 = _ddim_sample(z2, diffusion, cfg["phase2_ddim_steps"], cfg["ddim_eta"],
                              require_grad=True, **ddim_kwargs)
            v2_phys = _v_to_phys(v2.squeeze(0).squeeze(0)).clamp(VELOCITY_VMIN_M_S, VELOCITY_VMAX_M_S)
            loss = _obs_loss(forward_fn(v2_phys), target_seismic, kind=cfg["obs_loss"], mask=mask_dev)
            loss.backward()
            opt2.step()
            with torch.no_grad():
                v2_clip = v2.detach().clamp(-1.0, 1.0)
                rmse, mae, ssim = _metrics_norm(v2_clip, v_true_norm)
            history["rmse"].append(rmse)
            history["mae"].append(mae)
            history["ssim"].append(ssim)
            # phase2 里 velocity_pred = DDIM(z2)，v_gen ≡ velocity_pred；两份指标相同
            history["rmse_vgen"].append(rmse)
            history["mae_vgen"].append(mae)
            history["ssim_vgen"].append(ssim)
            history["obs_loss"].append(float(loss.item()))
            history["reg_loss"].append(0.0)
            history["total_loss"].append(float(loss.item()))

            if (p2_step + 1) in snap_p2_set:
                snap_data.append({
                    "phase": "phase2", "step": p2_step + 1,
                    "velocity_v": _capture_v_phys(v2_clip),
                    "velocity_vgen": _capture_v_phys(v2_clip),
                    "latent_z": _capture_latent(z2),
                    "rmse": rmse, "mae": mae, "ssim": ssim,
                    "rmse_vgen": rmse, "mae_vgen": mae, "ssim_vgen": ssim,
                })

        # 用 phase2 末态作为最终输出
        with torch.no_grad():
            v_final = _ddim_sample(z2, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                                   require_grad=False, **ddim_kwargs).clamp(-1.0, 1.0)
    else:
        v_final = v.detach().clamp(-1.0, 1.0)

    velocity_pred_phys = _v_to_phys(v_final.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="dlo_fwi",
        params=cfg,
        extra={"z_loss": z_hist, "lambda": lam_hist, "phase2_used": phase2_used,
               "snapshots": snap_data},
    )


# 旧名兼容别名（避免外部脚本立即崩；新代码请用 run_dlo_fwi）
run_method_b = run_dlo_fwi
METHOD_B_DEFAULTS = DLO_FWI_DEFAULTS


# =============================================================================
# 6b) DLO-FWI Adaptive — 数据驱动的 λ schedule
#     不再依赖 hard step counter（warmup/ramp），而是按 z_loss 的 EMA 自适应：
#
#         λ_eff(i) = λ_max · sigmoid((τ − L_z_ema(i)) / κ)
#
#     - L_z_ema 是 ‖DDIM(z) − v‖² 的指数滑动平均
#     - z 拟合还不好 (L_z_ema > τ) → λ_eff ≈ 0，引导被自动抑制
#     - z 拟合到位 (L_z_ema < τ) → λ_eff → λ_max，引导启动
#
#     等价于"数据驱动地决定何时开启 manifold consistency penalty"。
# =============================================================================
DLO_FWI_ADAPTIVE_DEFAULTS: Dict = {
    **DLO_FWI_DEFAULTS,
    # 自适应 λ 参数（覆盖原 warmup/ramp 字段，不再被读取）
    "lambda_tau":       0.02,   # sigmoid 中心点（z_loss EMA 的阈值）
    "lambda_kappa":     0.01,   # sigmoid 陡峭度（越小切换越突）
    "lambda_ema_alpha": 0.1,    # z_loss EMA 系数（越大越灵敏）
}


def run_dlo_fwi_adaptive(
    seismic_obs: torch.Tensor,
    velocity_true: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    diffusion: Optional[DiffusionPrior],
    device: torch.device,
    params: Optional[Dict] = None,
    trace_mask: Optional[torch.Tensor] = None,
) -> InversionResult:
    """**DLO-FWI Adaptive**：与 ``run_dlo_fwi`` 相同流程，但 λ 由 z_loss EMA 驱动。

    λ_eff(i) = λ_max · sigmoid( (τ − L_z_ema(i)) / κ )

    参数（覆盖 ``DLO_FWI_DEFAULTS`` 之上）：
      - ``lambda_max``       : λ 的上限（同 schedule 版）
      - ``lambda_tau``       : sigmoid 中心点（L_z_ema 降到此值时 λ_eff = λ_max/2）
      - ``lambda_kappa``     : sigmoid 陡峭度
      - ``lambda_ema_alpha`` : EMA 平滑系数（0~1，建议 0.05~0.3）

    其它（DDIM 步数、phase2、snapshot 等）与 ``run_dlo_fwi`` 一致。
    ``warmup_steps`` / ``ramp_steps`` 字段会被忽略。
    """
    if diffusion is None:
        raise ValueError("DLO-FWI-adaptive requires a DiffusionPrior.")
    cfg = {**DLO_FWI_ADAPTIVE_DEFAULTS, **(params or {})}

    ddim_kwargs = dict(
        clip_sample=bool(cfg.get("ddim_clip_sample", True)),
        clip_sample_range=float(cfg.get("ddim_clip_sample_range", 1.0)),
        use_clipped_model_output=bool(cfg.get("ddim_use_clipped_model_output", True)),
    )

    v_true_np = velocity_true.detach().cpu().numpy().astype(np.float32).squeeze()
    v_init_norm_np = _smoothed_init_norm(v_true_np, cfg["smooth_sigma"])
    v_true_norm = torch.from_numpy(((v_true_np - VELOCITY_CENTER_M_S) / VELOCITY_SCALE_M_S).astype(np.float32)).view(1, 1, *v_true_np.shape).to(device)
    target_seismic = seismic_obs.to(device)
    mask_dev = trace_mask.to(device) if trace_mask is not None else None

    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)

    H, W = v_true_np.shape
    v = torch.from_numpy(v_init_norm_np).view(1, 1, H, W).to(device).clamp(-1.0, 1.0).requires_grad_(True)
    z = torch.randn(1, 1, H, W, device=device, dtype=torch.float32, requires_grad=True)

    n_iters = int(cfg["n_iters"])
    opt_v = torch.optim.Adam([v], lr=cfg["lr_v"])
    opt_z = torch.optim.Adam([z], lr=cfg["lr_z"])

    lambda_max = float(cfg["lambda_max"])
    lambda_tau = float(cfg["lambda_tau"])
    lambda_kappa = max(float(cfg["lambda_kappa"]), 1e-12)
    ema_alpha = float(cfg["lambda_ema_alpha"])

    def _sigmoid(x: float) -> float:
        # 数值稳定的 sigmoid
        if x >= 0:
            ex = math.exp(-x)
            return 1.0 / (1.0 + ex)
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _lambda_from_ema(lz_ema: Optional[float]) -> float:
        if lz_ema is None:
            return 0.0
        return lambda_max * _sigmoid((lambda_tau - lz_ema) / lambda_kappa)

    history: Dict[str, List[float]] = {
        "rmse": [], "mae": [], "ssim": [],
        "obs_loss": [], "reg_loss": [], "total_loss": [],
    }
    z_hist: List[float] = []
    z_ema_hist: List[float] = []
    lam_hist: List[float] = []

    n_snap = int(cfg.get("snapshots", 0))
    phase2_iters_cfg = int(cfg.get("phase2_steps", 0))
    snap_p1_set = set(_make_snapshot_steps(n_iters, n_snap))
    snap_p2_set = set(_make_snapshot_steps(phase2_iters_cfg, n_snap)) if phase2_iters_cfg > 0 else set()
    snap_data: List[Dict] = []

    if 0 in snap_p1_set:
        with torch.no_grad():
            v_gen_init = _ddim_sample(z, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                                       require_grad=False, **ddim_kwargs)
            r0v, m0v, s0v = _metrics_norm(v, v_true_norm)
            r0g, m0g, s0g = _metrics_norm(v_gen_init, v_true_norm)
        snap_data.append({
            "phase": "phase1", "step": 0,
            "velocity_v": _capture_v_phys(v),
            "velocity_vgen": _capture_v_phys(v_gen_init),
            "latent_z": _capture_latent(z),
            "rmse": r0v, "mae": m0v, "ssim": s0v,
            "rmse_vgen": r0g, "mae_vgen": m0g, "ssim_vgen": s0g,
        })

    z_loss_ema: Optional[float] = None

    for step in range(n_iters):
        # ── Step 1: update z ────────────────────────────────────────────────
        loss_z_val = 0.0
        for _ in range(int(cfg["z_steps_per_iter"])):
            opt_z.zero_grad(set_to_none=True)
            v_gen_z = _ddim_sample(z, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                                   require_grad=True, **ddim_kwargs)
            loss_z = F.mse_loss(v_gen_z, v.detach())
            loss_z.backward()
            opt_z.step()
            loss_z_val = float(loss_z.item())

        # 更新 z_loss EMA + 计算自适应 λ
        if z_loss_ema is None:
            z_loss_ema = loss_z_val
        else:
            z_loss_ema = ema_alpha * loss_z_val + (1.0 - ema_alpha) * z_loss_ema
        lam = _lambda_from_ema(z_loss_ema)

        # ── Step 2: decode v_gen ─────────────────────────────────────────────
        v_gen = _ddim_sample(z, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                             require_grad=False, **ddim_kwargs)

        # ── Step 3: update v ─────────────────────────────────────────────────
        opt_v.zero_grad(set_to_none=True)
        v_phys = _v_to_phys(v.squeeze(0).squeeze(0)).clamp(VELOCITY_VMIN_M_S, VELOCITY_VMAX_M_S)
        loss_obs = _obs_loss(forward_fn(v_phys), target_seismic, kind=cfg["obs_loss"], mask=mask_dev)
        if lam > 0:
            loss_guide = F.mse_loss(v, v_gen.detach())
            loss = loss_obs + lam * loss_guide
            reg_log = float(loss_guide.item())
        else:
            loss = loss_obs
            reg_log = 0.0
        loss.backward()
        opt_v.step()
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        # history 统一记录 R(z) 的指标（与 phase2 一致，保证曲线全程同语义）
        with torch.no_grad():
            rmse, mae, ssim = _metrics_norm(v_gen, v_true_norm)
            rv, mv, sv = _metrics_norm(v, v_true_norm)
        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["ssim"].append(ssim)
        history["obs_loss"].append(float(loss_obs.item()))
        history["reg_loss"].append(reg_log)
        history["total_loss"].append(float(loss.item()))
        z_hist.append(loss_z_val)
        z_ema_hist.append(float(z_loss_ema))
        lam_hist.append(lam)

        if (step + 1) in snap_p1_set:
            snap_data.append({
                "phase": "phase1", "step": step + 1,
                "velocity_v": _capture_v_phys(v),
                "velocity_vgen": _capture_v_phys(v_gen),
                "latent_z": _capture_latent(z),
                "rmse": rv, "mae": mv, "ssim": sv,
                "rmse_vgen": rmse, "mae_vgen": mae, "ssim_vgen": ssim,
            })

    # ── Phase 2（与 run_dlo_fwi 一致） ────────────────────────────────────────
    phase2_iters = int(cfg.get("phase2_steps", 0))
    phase2_used = False
    if phase2_iters > 0:
        phase2_used = True
        z2 = z.detach().clone().requires_grad_(True)
        opt2 = torch.optim.Adam([z2], lr=cfg["phase2_lr"])

        if 0 in snap_p2_set:
            with torch.no_grad():
                v2_init = _ddim_sample(z2, diffusion, cfg["phase2_ddim_steps"],
                                       cfg["ddim_eta"], require_grad=False,
                                       **ddim_kwargs).clamp(-1.0, 1.0)
                r2, m2, s2 = _metrics_norm(v2_init, v_true_norm)
            snap_data.append({
                "phase": "phase2", "step": 0,
                "velocity_v": _capture_v_phys(v2_init),
                "velocity_vgen": _capture_v_phys(v2_init),
                "latent_z": _capture_latent(z2),
                "rmse": r2, "mae": m2, "ssim": s2,
                "rmse_vgen": r2, "mae_vgen": m2, "ssim_vgen": s2,
            })

        for p2_step in range(phase2_iters):
            opt2.zero_grad(set_to_none=True)
            v2 = _ddim_sample(z2, diffusion, cfg["phase2_ddim_steps"], cfg["ddim_eta"],
                              require_grad=True, **ddim_kwargs)
            v2_phys = _v_to_phys(v2.squeeze(0).squeeze(0)).clamp(VELOCITY_VMIN_M_S, VELOCITY_VMAX_M_S)
            loss = _obs_loss(forward_fn(v2_phys), target_seismic, kind=cfg["obs_loss"], mask=mask_dev)
            loss.backward()
            opt2.step()
            with torch.no_grad():
                v2_clip = v2.detach().clamp(-1.0, 1.0)
                rmse, mae, ssim = _metrics_norm(v2_clip, v_true_norm)
            history["rmse"].append(rmse)
            history["mae"].append(mae)
            history["ssim"].append(ssim)
            history["obs_loss"].append(float(loss.item()))
            history["reg_loss"].append(0.0)
            history["total_loss"].append(float(loss.item()))

            if (p2_step + 1) in snap_p2_set:
                snap_data.append({
                    "phase": "phase2", "step": p2_step + 1,
                    "velocity_v": _capture_v_phys(v2_clip),
                    "velocity_vgen": _capture_v_phys(v2_clip),
                    "latent_z": _capture_latent(z2),
                    "rmse": rmse, "mae": mae, "ssim": ssim,
                    "rmse_vgen": rmse, "mae_vgen": mae, "ssim_vgen": ssim,
                })

        with torch.no_grad():
            v_final = _ddim_sample(z2, diffusion, cfg["ddim_steps"], cfg["ddim_eta"],
                                   require_grad=False, **ddim_kwargs).clamp(-1.0, 1.0)
    else:
        v_final = v.detach().clamp(-1.0, 1.0)

    velocity_pred_phys = _v_to_phys(v_final.squeeze().detach().cpu().numpy())
    velocity_init_phys = _v_to_phys(v_init_norm_np)
    return InversionResult(
        velocity_pred=velocity_pred_phys.astype(np.float32),
        velocity_init=velocity_init_phys.astype(np.float32),
        history=history,
        method="dlo_fwi_adaptive",
        params=cfg,
        extra={"z_loss": z_hist, "z_loss_ema": z_ema_hist, "lambda": lam_hist,
               "phase2_used": phase2_used, "snapshots": snap_data},
    )


# =============================================================================
# 注册表
# =============================================================================
METHOD_REGISTRY: Dict[str, Callable] = {
    "tikhonov": run_tikhonov,
    "tv": run_tv,
    "red_diffeq": run_red_diffeq,
    "diffusion_fwi": run_diffusion_fwi,
    "diffusion_ilvr": run_diffusion_ilvr,
    "dlo_fwi": run_dlo_fwi,
    "dlo_fwi_adaptive": run_dlo_fwi_adaptive,
    "method_b": run_dlo_fwi,        # alias，新代码请用 'dlo_fwi'
}
