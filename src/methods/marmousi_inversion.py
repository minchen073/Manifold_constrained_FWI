"""Marmousi 70x190 FWI inversion with four regularizers.

Uses our parametric cupy forward operator and our repo's OpenFWIUNetWrapper
DDIM checkpoints in ``pretrained_model/`` (NOT the four-in-one RED-DiffEq model).

Methods:
  - physical:    no regularization (pure data fidelity)
  - tikhonov:    L2 of spatial gradient  -- benchmark.tikhonov_loss
  - tv:          L1 of spatial gradient  -- benchmark.total_variation_loss
  - red-diffeq:  RED-DiffEq diffusion-model regularizer applied to overlapping
                 70x70 patches because the network was trained at 70x70.

mu is the unpadded normalized velocity (1, 1, 70, 190) in [-1, 1].
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

MANIFOLD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MANIFOLD_ROOT))
sys.path.insert(0, str(MANIFOLD_ROOT / "training"))

from src.seismic.wave_equation_forward_parametric import WaveEquationForward


# =============================================================================
# Normalization helpers (m/s <-> [-1, 1], matches RED-DiffEq's v_normalize)
# =============================================================================

V_MIN, V_MAX = 1500.0, 4500.0
V_CENTER, V_SCALE = 3000.0, 1500.0  # (V_MIN + V_MAX)/2,  (V_MAX - V_MIN)/2


def v_norm(v_phys):
    return (v_phys - V_CENTER) / V_SCALE


def v_denorm(v_n):
    return v_n * V_SCALE + V_CENTER


# =============================================================================
# Forward-operator wrapper (cupy engine -> torch nn.Module interface)
# =============================================================================

class WaveEqFWIForward(nn.Module):
    """Input (B,1,nz,nx) normalized -> output (B, ns, nt, ng) seismic."""

    def __init__(self, nz: int, ctx: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.nz = int(nz)
        self.nx = int(ctx["n_grid"])
        self.ns = int(ctx["ns"])
        self.nt = int(ctx["nt"])
        self.dx = float(ctx["dx"])
        sz_grid = int(round(ctx["sz"] / ctx["dx"]))
        gz_grid = int(round(ctx["gz"] / ctx["dx"]))
        self.engine = WaveEquationForward(
            nz=self.nz, nx=self.nx, ns=self.ns, nt=self.nt,
            dx=self.dx, dt=float(ctx["dt"]), freq=float(ctx["f"]),
            nbc=int(ctx["nbc"]),
            source_z_grid=sz_grid, recv_z_grid=gz_grid,
        )

    def forward(self, v_normalized: torch.Tensor) -> torch.Tensor:
        if v_normalized.dim() != 4 or v_normalized.shape[1] != 1:
            raise ValueError(f"expected (B,1,H,W), got {tuple(v_normalized.shape)}")
        v_phys = v_denorm(v_normalized).clamp(V_MIN, V_MAX)
        outs = [self.engine(v_phys[b, 0]) for b in range(v_phys.shape[0])]
        return torch.stack(outs, dim=0)


# =============================================================================
# Diffusion prior loader (uses our OpenFWIUNetWrapper)
# =============================================================================

@dataclass
class DiffusionPrior:
    wrapper: nn.Module           # OpenFWIUNetWrapper (input 70x70)
    alphas_cumprod: torch.Tensor # (T,)
    num_train_timesteps: int
    patch_size: int              # = wrapper.spatial = 70
    final_alpha_cumprod: float = 1.0  # DDIM t=-1 boundary

    def denoise(self, x_t: torch.Tensor, t) -> torch.Tensor:
        """eps_theta(x_t, t). t may be int or (B,) long tensor."""
        if isinstance(t, int):
            t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        else:
            t_tensor = t.to(device=x_t.device, dtype=torch.long)
        return self.wrapper(x_t, t_tensor).sample


def _build_alphas_cumprod_from_scheduler_config(cfg: dict) -> torch.Tensor:
    """Build alphas_cumprod from a diffusers DDPMScheduler-style config."""
    T = int(cfg["num_train_timesteps"])
    schedule = cfg.get("beta_schedule", "linear")
    if schedule == "linear":
        betas = torch.linspace(float(cfg["beta_start"]), float(cfg["beta_end"]), T,
                                dtype=torch.float64)
    elif schedule == "scaled_linear":
        betas = (torch.linspace(float(cfg["beta_start"]) ** 0.5,
                                 float(cfg["beta_end"]) ** 0.5, T, dtype=torch.float64) ** 2)
    else:
        raise ValueError(f"unsupported beta_schedule: {schedule}")
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).float()


def load_diffusion_prior(name_or_path, device: torch.device,
                          pretrained_root: Optional[Path] = None) -> DiffusionPrior:
    """Load one of pretrained_model/<NAME>_DDIM/{model.pt,scheduler_config.json}.

    ``name_or_path`` can be a bare name (resolved against pretrained_root) or a
    direct path to a checkpoint directory.
    """
    # silence diffusers stderr on import (custom-op warnings)
    import os
    _null = os.open(os.devnull, os.O_WRONLY); _saved = os.dup(2)
    os.dup2(_null, 2); os.close(_null)
    try:
        from openfwi_unet_wrapper import load_openfwi_checkpoint
    finally:
        os.dup2(_saved, 2); os.close(_saved)

    import json
    p = Path(name_or_path)
    if not p.is_absolute():
        root = pretrained_root or (MANIFOLD_ROOT / "pretrained_model")
        cand = root / p
        if not cand.exists() and not (cand.name.endswith("_DDIM")):
            cand = root / f"{p}_DDIM"
        p = cand
    if not p.exists():
        raise FileNotFoundError(f"diffusion checkpoint not found: {p}")
    model_pt = p / "model.pt" if p.is_dir() else p
    sched_json = p / "scheduler_config.json" if p.is_dir() else p.parent / "scheduler_config.json"
    if not model_pt.is_file():
        raise FileNotFoundError(f"model.pt not found at {model_pt}")
    if not sched_json.is_file():
        raise FileNotFoundError(f"scheduler_config.json not found at {sched_json}")

    wrapper = load_openfwi_checkpoint(model_pt, map_location="cpu", torch_dtype=torch.float32)
    wrapper = wrapper.to(device).eval()
    cfg = json.loads(sched_json.read_text())
    abar = _build_alphas_cumprod_from_scheduler_config(cfg).to(device)
    return DiffusionPrior(
        wrapper=wrapper, alphas_cumprod=abar,
        num_train_timesteps=int(cfg["num_train_timesteps"]),
        patch_size=int(wrapper.spatial),
    )


# =============================================================================
# Regularizers
# =============================================================================

def tikhonov_loss(mu: torch.Tensor) -> torch.Tensor:
    """L2 of spatial gradient, mean per-sample. (matches RED-DiffEq benchmark.)"""
    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :]
    return ((diff_x ** 2).mean(dim=(1, 2, 3))
            + (diff_y ** 2).mean(dim=(1, 2, 3)))


def total_variation_loss(mu: torch.Tensor) -> torch.Tensor:
    diff_x = (mu[:, :, :, 1:] - mu[:, :, :, :-1]).abs()
    diff_y = (mu[:, :, 1:, :] - mu[:, :, :-1, :]).abs()
    return diff_x.mean(dim=(1, 2, 3)) + diff_y.mean(dim=(1, 2, 3))


def _patch_positions(width: int, patch: int):
    """Overlapping patches covering [0, width] with patch size ``patch``.

    Mirrors RED-DiffEq's calculate_patches: k = ceil(W / patch), stride = (W - patch)/(k - 1).
    Returns ([(s, e), ...], [overlap_widths, ...]).
    """
    if width <= patch:
        return [(0, width)], []
    k = math.ceil(width / patch)
    s_stride = (width - patch) / (k - 1)
    positions = []
    for i in range(k):
        if i == k - 1:
            positions.append((width - patch, width))
        else:
            start = int(i * s_stride)
            positions.append((start, min(start + patch, width)))
    overlaps = [positions[i][1] - positions[i + 1][0] for i in range(k - 1)]
    return positions, overlaps


# =============================================================================
# RED-DiffEq four-in-one prior (model-4.pt, GaussianDiffusion at 72x72)
# =============================================================================

def load_reddiffeq_model(ckpt_path, device: torch.device,
                          dim: int = 64, dim_mults=(1, 2, 4, 8),
                          channels: int = 1, image_size: int = 72,
                          timesteps: int = 1000, sampling_timesteps: int = 250,
                          objective: str = "pred_noise"):
    """Build RED-DiffEq's GaussianDiffusion+Unet and load model-4.pt weights.

    Returns the GaussianDiffusion module (in eval mode, on device). Requires
    ``repo/red-diffeq`` on sys.path; the Accelerator wrapping is skipped — we
    only need the eval-time interface (q_sample, model_predictions, etc.).
    """
    from red_diffeq.models.diffusion import Unet, GaussianDiffusion

    model = Unet(dim=dim, dim_mults=tuple(dim_mults), flash_attn=False,
                 channels=channels)
    diffusion = GaussianDiffusion(
        model, image_size=image_size, timesteps=timesteps,
        sampling_timesteps=sampling_timesteps, objective=objective,
    ).to(device)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"RED-DiffEq checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    diffusion.load_state_dict(state)
    diffusion.eval()
    return diffusion


def reddiffeq_orig_loss(mu: torch.Tensor, diffusion_model, *,
                         use_time_weight: bool = False, sigma_x0: float = 1e-4,
                         max_t: Optional[int] = None,
                         generator: Optional[torch.Generator] = None):
    """RED regularization using RED-DiffEq's own GaussianDiffusion model.

    Pads mu (1,1,H,W) to (1,1,H+2,W+2) so that ``diffusion_crop`` inside
    ``get_reg_loss_patched`` recovers (H, W). The patched loss splits across
    W into 70-wide chunks (so the internal patches are 72-wide after pad).

    Returns (reg_loss_per_sample (B,), t_used (B,)).
    """
    from red_diffeq.regularization.diffusion import RED_DiffEq
    import torch.nn.functional as F

    if sigma_x0 > 0:
        x0_pred = mu + sigma_x0 * torch.randn(mu.shape, device=mu.device,
                                                dtype=mu.dtype, generator=generator)
    else:
        x0_pred = mu

    mu_padded = F.pad(x0_pred, (1, 1, 1, 1), mode="constant", value=0)

    red = RED_DiffEq(
        diffusion_model, use_time_weight=use_time_weight,
        sigma_x0=0.0,  # we already applied sigma_x0 above, so disable inside
        fixed_timestep=max_t,
    )
    reg_per_model, _, t_used = red.get_reg_loss_patched(mu_padded, generator=generator)
    return reg_per_model, t_used


def red_diffeq_loss(mu: torch.Tensor, prior: DiffusionPrior, *,
                     sigma_x0: float = 1e-4, max_t: Optional[int] = None,
                     use_time_weight: bool = False,
                     generator: Optional[torch.Generator] = None):
    """RED-DiffEq regularization on overlapping patches.

    Returns (reg_loss_per_sample (B,), t_used (B,)). Computation matches
    red_diffeq.regularization.diffusion.RED_DiffEq.get_reg_loss_patched but
    with our OpenFWIUNetWrapper instead of GaussianDiffusion.
    """
    if sigma_x0 > 0:
        x0_pred = mu + sigma_x0 * torch.randn(mu.shape, device=mu.device,
                                                dtype=mu.dtype, generator=generator)
    else:
        x0_pred = mu

    B = mu.shape[0]
    H, W = mu.shape[2], mu.shape[3]
    P = prior.patch_size
    if H != P:
        raise ValueError(f"height {H} must equal patch size {P}")

    T_max = max_t if max_t is not None else prior.num_train_timesteps
    t = torch.randint(0, T_max, (B,), generator=generator,
                       device=mu.device, dtype=torch.long)
    abar = prior.alphas_cumprod[t].to(mu.device).float().view(B, 1, 1, 1)

    noise_full = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)

    grad_field = torch.zeros_like(mu)
    weight_map = torch.zeros_like(mu)

    positions, overlaps = _patch_positions(W, P)

    for k, (s, e) in enumerate(positions):
        x0_patch = x0_pred[:, :, :, s:e]
        eps_patch = noise_full[:, :, :, s:e]
        x_t_patch = abar.sqrt() * x0_patch + (1.0 - abar).sqrt() * eps_patch

        with torch.no_grad():
            eps_pred = prior.wrapper(x_t_patch.detach(), t).sample
            # x0 clipping + rederive (matches RED-DiffEq's clip_x_start=True, rederive_pred_noise=True)
            x0_hat = ((x_t_patch - (1.0 - abar).sqrt() * eps_pred) / abar.sqrt()).clamp(-1.0, 1.0)
            eps_pred = (x_t_patch - abar.sqrt() * x0_hat) / (1.0 - abar).sqrt()

        grad_patch = (eps_pred - eps_patch).detach()

        w = torch.ones(e - s, device=mu.device, dtype=mu.dtype)
        if k > 0:
            w[: overlaps[k - 1]] = 0.5
        if k < len(positions) - 1:
            w[-overlaps[k]:] = 0.5
        w = w.view(1, 1, 1, -1)

        grad_field[:, :, :, s:e] += grad_patch * w
        weight_map[:, :, :, s:e] += w

    grad_field = grad_field / weight_map.clamp(min=1e-8)

    if use_time_weight:
        wt = ((1.0 - abar) / abar.clamp(min=1e-8)).sqrt()
        grad_field = grad_field * wt

    reg_field = grad_field * x0_pred
    return reg_field.mean(dim=(1, 2, 3)), t


# =============================================================================
# Initial model (smoothed normalized truth)
# =============================================================================

def smoothed_initial_norm(v_true_phys: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smoothed normalized initial model. Input/output (H, W)."""
    v_n = v_norm(v_true_phys.astype(np.float32))
    return gaussian_filter(v_n, sigma=sigma).astype(np.float32)


# =============================================================================
# SSIM (in-house, no external dep)
# =============================================================================

def _ssim_window(window_size: int, channel: int, device, dtype):
    sigma = 1.5
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
                            for x in range(window_size)], device=device, dtype=dtype)
    gauss = gauss / gauss.sum()
    win_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return win_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM between two (B,C,H,W) tensors in [0, 1]. Returns per-sample SSIM (B,)."""
    C = img1.shape[1]
    win = _ssim_window(window_size, C, img1.device, img1.dtype)
    pad = window_size // 2
    import torch.nn.functional as F
    mu1 = F.conv2d(img1, win, padding=pad, groups=C)
    mu2 = F.conv2d(img2, win, padding=pad, groups=C)
    mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, win, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, win, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
                / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
    return ssim_map.mean(dim=(1, 2, 3))


# =============================================================================
# DDIM sampler + patch blending (for DLO)
# =============================================================================

def _build_ddim_timesteps(num_steps: int, num_train: int) -> List[int]:
    if num_steps <= 1:
        return [num_train - 1]
    step = num_train / num_steps
    return [int(round((num_steps - 1 - i) * step)) for i in range(num_steps)]


def _ddim_sample(z: torch.Tensor, prior: DiffusionPrior, num_steps: int,
                  eta: float = 0.0, require_grad: bool = True,
                  clip_sample: bool = True, clip_sample_range: float = 1.0,
                  use_clipped_model_output: bool = True) -> torch.Tensor:
    """DDIM reverse sampling z -> x_0. Mirrors diffusers DDIMScheduler.step."""
    timesteps = _build_ddim_timesteps(num_steps, prior.num_train_timesteps)
    abar = prior.alphas_cumprod.to(z.device).float()

    def _step():
        x = z
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            eps = prior.denoise(x, int(t))
            abar_t = abar[int(t)]
            abar_prev = (abar[int(t_prev)] if t_prev >= 0
                          else torch.tensor(prior.final_alpha_cumprod,
                                             device=z.device, dtype=torch.float32))
            pred_x0 = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt()
            if clip_sample:
                pred_x0 = pred_x0.clamp(-clip_sample_range, clip_sample_range)
            variance = ((1.0 - abar_prev) / (1.0 - abar_t)
                        * (1.0 - abar_t / abar_prev)).clamp(min=0.0)
            sigma = eta * variance.sqrt()
            eps_used = eps
            if use_clipped_model_output:
                eps_used = (x - abar_t.sqrt() * pred_x0) / (1.0 - abar_t).sqrt()
            coef_dir = (1.0 - abar_prev - sigma ** 2).clamp(min=0.0).sqrt()
            x = abar_prev.sqrt() * pred_x0 + coef_dir * eps_used
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    if require_grad:
        return _step()
    with torch.no_grad():
        return _step()


def _patch_blend(parts: List[torch.Tensor], positions: List, overlaps: List,
                  W: int) -> torch.Tensor:
    """Blend patch tensors (each (B,1,H,P)) into (B,1,H,W) with 0.5 weights in
    overlap regions — matches the weighting used in red_diffeq_loss.
    """
    sample = parts[0]
    out = torch.zeros(sample.shape[0], sample.shape[1], sample.shape[2], W,
                       device=sample.device, dtype=sample.dtype)
    wmap = torch.zeros_like(out)
    for k, ((s, e), patch) in enumerate(zip(positions, parts)):
        w = torch.ones(e - s, device=sample.device, dtype=sample.dtype)
        if k > 0:
            w[: overlaps[k - 1]] = 0.5
        if k < len(positions) - 1:
            w[-overlaps[k]:] = 0.5
        w4 = w.view(1, 1, 1, -1)
        out[:, :, :, s:e] += patch * w4
        wmap[:, :, :, s:e] += w4
    return out / wmap.clamp(min=1e-8)


# =============================================================================
# Method config (hyperparameter defaults, mirrors RED-DiffEq Marmousi config)
# =============================================================================

SHARED_DEFAULTS: Dict = dict(
    ts=300, lr=0.03, sigma_init=20.0, seed=8888,
    noise_std=0.0, noise_type="gaussian", missing_number=0,
    cosine_lr=True, clamp=True,
)

# DLO (Decoupled Latent Optimization) = Phase 1 of run_dataset_average.py
# METHOD_PARAMS["dlo_fwi"]: alternating z/v updates with soft coupling.
DLO_DEFAULTS: Dict = dict(
    n_iters=300, lr_v=0.03, lr_z=0.02, z_steps_per_iter=1,
    sigma_init=20.0,                # smooth_sigma in averaged script
    lambda_max=0.5, warmup_steps=0, ramp_steps=0,
    ddim_steps=3, ddim_eta=0.0,
    ddim_clip_sample=True, ddim_clip_sample_range=1.0,
    ddim_use_clipped_model_output=True,
    seed=8888, clamp=True, obs_loss="l1",
)

METHOD_DEFAULTS: Dict[str, Dict] = {
    "physical":   {"reg_lambda": 0.0},
    "tikhonov":   {"reg_lambda": 0.01},
    "tv":         {"reg_lambda": 0.01},
    "red-diffeq": {"reg_lambda": 0.75, "sigma_x0": 1e-4, "max_t": None,
                   "use_time_weight": False},
    "red-diffeq-orig": {"reg_lambda": 0.75, "sigma_x0": 1e-4, "max_t": None,
                         "use_time_weight": False},
    # DLO uses its own loop (run_dlo_inversion). The dict is just a marker so
    # the driver's --methods choices recognize it; real defaults are DLO_DEFAULTS.
    "dlo":        {},
    # DiffusionFWI uses run_diffusion_fwi_inversion. Marker only; real defaults
    # are DIFFUSION_FWI_DEFAULTS.
    "diffusion-fwi": {}
}


# =============================================================================
# DiffusionFWI defaults (mirror inversion_methods.DIFFUSION_FWI_DEFAULTS)
# =============================================================================

DIFFUSION_FWI_DEFAULTS: Dict = dict(
    init_time_step=900,
    fwi_iters_per_step=10, lr=0.01, sigma_init=20.0,
    obs_loss="l1",
    optim="adam", use_scheduler=False, grad_clip=1.0,
    grad_smooth_sigma=0, grad_smooth_sigma_v=0, grad_smooth_kernel=5,
    velocity_blur_kernel=3, velocity_blur_sigma=0.3,
    grad_normalize=True,
    vel_blur_sigma=0.0, vel_blur_kernel=3,
    ddpm_noise_scale=0.001,
    seed=8888,
)


# =============================================================================
# Helpers: noise & missing-trace (kept minimal, no external dep)
# =============================================================================

def add_noise(y: torch.Tensor, std: float, kind: str = "gaussian",
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if std == 0:
        return y
    if kind == "gaussian":
        noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype) * std
    elif kind == "laplace":
        u = torch.rand(y.shape, generator=generator, device=y.device, dtype=y.dtype) - 0.5
        noise = -std * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    return y + noise


def missing_trace_mask(y: torch.Tensor, num_missing: int,
                        generator: Optional[torch.Generator] = None):
    """Zero out ``num_missing`` random receivers (shared across shots per batch)."""
    mask = torch.ones_like(y)
    if num_missing == 0:
        return y, mask
    B, S, T, R = y.shape
    out = y.clone()
    for b in range(B):
        idx = torch.randperm(R, generator=generator, device=y.device)[:num_missing]
        out[b, :, :, idx] = 0
        mask[b, :, :, idx] = 0
    return out, mask


# =============================================================================
# Result container
# =============================================================================

@dataclass
class InversionResult:
    velocity_pred_phys: np.ndarray
    velocity_init_phys: np.ndarray
    velocity_true_phys: np.ndarray
    history: Dict[str, List[float]] = field(default_factory=dict)
    method: str = ""
    params: Dict = field(default_factory=dict)


# =============================================================================
# Inversion loop
# =============================================================================

def run_inversion(
    method: str,
    seismic_obs: torch.Tensor,
    velocity_true_phys: torch.Tensor,
    forward_op: WaveEqFWIForward,
    device: torch.device,
    diffusion_prior: Optional[DiffusionPrior] = None,
    reddiffeq_orig_model=None,
    params: Optional[dict] = None,
    progress: bool = True,
    log_interval: int = 10,
) -> InversionResult:
    """Run one inversion. method in METHOD_DEFAULTS keys."""
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"unknown method '{method}'")

    cfg = {**SHARED_DEFAULTS, **METHOD_DEFAULTS[method]}
    if params:
        cfg.update(params)
    if method == "red-diffeq" and diffusion_prior is None:
        raise ValueError("red-diffeq requires diffusion_prior")
    if method == "red-diffeq-orig" and reddiffeq_orig_model is None:
        raise ValueError("red-diffeq-orig requires reddiffeq_orig_model")

    g = torch.Generator(device=device).manual_seed(int(cfg["seed"]))

    seismic_obs = seismic_obs.to(device).float()
    vt = velocity_true_phys.float()                        # CPU (1,1,H,W)
    vt_dev = vt.to(device)

    # initial model
    init_n_np = smoothed_initial_norm(vt[0, 0].cpu().numpy(), cfg["sigma_init"])
    mu = torch.from_numpy(init_n_np).view(1, 1, *init_n_np.shape).to(device)
    mu = mu.clone().detach().requires_grad_(True)

    # Observation is expected to already include any noise / missing-trace
    # mask applied at the driver level (so all methods see the same data).
    y = seismic_obs
    mask = torch.ones_like(y)

    # optimizer
    optimizer = torch.optim.Adam([mu], lr=cfg["lr"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["ts"], eta_min=0.0)
                  if cfg["cosine_lr"] else None)

    history = {k: [] for k in ("total_loss", "obs_loss", "reg_loss",
                                  "ssim", "mae", "rmse")}

    pbar_iter = range(cfg["ts"])
    if progress:
        from tqdm.auto import tqdm
        pbar_iter = tqdm(pbar_iter, desc=f"[{method}]", unit="step")

    for step in pbar_iter:
        # forward
        pred = forward_op(mu)

        # obs loss: masked L1
        per = (pred.float() - y.float()).abs() * mask
        denom = mask.sum(dim=tuple(range(1, mask.dim()))).clamp(min=1.0)
        obs_loss = per.sum(dim=tuple(range(1, per.dim()))) / denom

        # reg loss
        t_used = None
        if method == "physical":
            reg_loss = torch.zeros_like(obs_loss)
        elif method == "tikhonov":
            reg_loss = tikhonov_loss(mu)
        elif method == "tv":
            reg_loss = total_variation_loss(mu)
        elif method == "red-diffeq":
            reg_loss, t_used = red_diffeq_loss(
                mu, diffusion_prior,
                sigma_x0=cfg["sigma_x0"], max_t=cfg["max_t"],
                use_time_weight=cfg["use_time_weight"], generator=g,
            )
        elif method == "red-diffeq-orig":
            reg_loss, t_used = reddiffeq_orig_loss(
                mu, reddiffeq_orig_model,
                sigma_x0=cfg["sigma_x0"], max_t=cfg["max_t"],
                use_time_weight=cfg["use_time_weight"], generator=g,
            )

        total_loss = obs_loss + cfg["reg_lambda"] * reg_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.sum().backward()
        optimizer.step()
        if cfg["clamp"]:
            with torch.no_grad():
                mu.data.clamp_(-1.0, 1.0)
        if scheduler is not None:
            scheduler.step()

        # metrics in normalized domain
        with torch.no_grad():
            mu_n = mu.detach()
            vt_n = v_norm(vt_dev)
            mae = (mu_n - vt_n).abs().mean(dim=(1, 2, 3))
            rmse = ((mu_n - vt_n) ** 2).mean(dim=(1, 2, 3)).sqrt()
            ssim_val = ssim(((mu_n + 1) / 2).clamp(0, 1), ((vt_n + 1) / 2).clamp(0, 1))

        history["total_loss"].append(float(total_loss.detach().mean().cpu()))
        history["obs_loss"].append(float(obs_loss.detach().mean().cpu()))
        history["reg_loss"].append(float(reg_loss.detach().mean().cpu()))
        history["mae"].append(float(mae.mean().cpu()))
        history["rmse"].append(float(rmse.mean().cpu()))
        history["ssim"].append(float(ssim_val.mean().cpu()))

        if progress and (step % log_interval == 0 or step == cfg["ts"] - 1):
            postfix = {"MAE": history["mae"][-1], "RMSE": history["rmse"][-1],
                       "SSIM": history["ssim"][-1]}
            if t_used is not None:
                postfix["t"] = int(t_used.float().mean().item())
            pbar_iter.set_postfix(postfix)

    with torch.no_grad():
        v_pred_phys = v_denorm(mu.detach())[0, 0].cpu().numpy()
        v_init_phys = v_denorm(torch.from_numpy(init_n_np)).numpy()
    return InversionResult(
        velocity_pred_phys=v_pred_phys,
        velocity_init_phys=v_init_phys,
        velocity_true_phys=vt[0, 0].cpu().numpy(),
        history=history,
        method=method,
        params=cfg,
    )


# =============================================================================
# DLO-FWI for 70x190 (3 patched z latents, 0.5-weight overlap blend)
# =============================================================================

def run_dlo_inversion(
    seismic_obs: torch.Tensor,
    velocity_true_phys: torch.Tensor,
    forward_op: WaveEqFWIForward,
    device: torch.device,
    diffusion_prior: DiffusionPrior,
    params: Optional[dict] = None,
    progress: bool = True,
    log_interval: int = 10,
) -> InversionResult:
    """Decoupled Latent Optimization FWI on 70x190.

    Splits 190 wide into 3 overlapping 70x70 patches (positions [(0,70),
    (60,130), (120,190)], overlaps [10, 10]); maintains an independent
    z_i for each patch and a global physical v in (1,1,70,190). Overlap
    blending uses 0.5/0.5 weights (matches red_diffeq_loss).

    Every outer step:
      (a) for each z_i: ``z_steps_per_iter`` updates of
          min ||DDIM(z_i) - v[:,:,:,s_i:e_i].detach()||^2
      (b) decode v_gen = blend(DDIM(z_0), DDIM(z_1), DDIM(z_2))  (no grad)
      (c) one Adam step on v:  L_wave(v) + lambda(step) * ||v - v_gen||^2

    Hyperparameter defaults mirror METHOD_PARAMS["dlo_fwi"] in
    scripts/run_dataset_average.py (Phase 1 only).
    """
    cfg = {**DLO_DEFAULTS, **(params or {})}
    if diffusion_prior is None:
        raise ValueError("DLO requires diffusion_prior")

    ddim_kw = dict(
        clip_sample=bool(cfg["ddim_clip_sample"]),
        clip_sample_range=float(cfg["ddim_clip_sample_range"]),
        use_clipped_model_output=bool(cfg["ddim_use_clipped_model_output"]),
    )

    g = torch.Generator(device=device).manual_seed(int(cfg["seed"]))

    seismic_obs = seismic_obs.to(device).float()
    vt = velocity_true_phys.float()
    vt_dev = vt.to(device)
    vt_n = v_norm(vt_dev)

    H, W = vt.shape[2], vt.shape[3]
    P = diffusion_prior.patch_size
    if H != P:
        raise ValueError(f"DLO expects height == patch_size ({P}), got {H}")
    positions, overlaps = _patch_positions(W, P)
    n_patches = len(positions)

    # smoothed initial v
    init_n_np = smoothed_initial_norm(vt[0, 0].cpu().numpy(), cfg["sigma_init"])
    v = torch.from_numpy(init_n_np).view(1, 1, H, W).to(device).clamp(-1.0, 1.0)
    v = v.clone().detach().requires_grad_(True)

    # patched latents z_i (each 70x70)
    z_list = [torch.randn(1, 1, P, P, device=device, dtype=torch.float32,
                            generator=g).requires_grad_(True)
              for _ in range(n_patches)]

    opt_v = torch.optim.Adam([v], lr=cfg["lr_v"])
    opt_z = torch.optim.Adam(z_list, lr=cfg["lr_z"])
    sched_v = (torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt_v, T_max=int(cfg["n_iters"]), eta_min=0.0)
               if cfg.get("use_scheduler", False) else None)

    def _lambda(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return 0.0
        prog = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["ramp_steps"]))
        return float(cfg["lambda_max"]) * prog

    history = {k: [] for k in ("total_loss", "obs_loss", "reg_loss",
                                 "ssim", "mae", "rmse",
                                 "rmse_vgen", "mae_vgen", "ssim_vgen", "z_loss")}

    n_iters = int(cfg["n_iters"])
    pbar = range(n_iters)
    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(pbar, desc="[dlo:p1]", unit="step")

    for step in pbar:
        # ── (a) update each z_i against the corresponding v_patch ─────────────
        loss_z_sum = 0.0
        for _ in range(int(cfg["z_steps_per_iter"])):
            opt_z.zero_grad(set_to_none=True)
            total_lz = 0.0
            for i, (s, e) in enumerate(positions):
                v_patch = v[:, :, :, s:e].detach()
                v_gen_i = _ddim_sample(z_list[i], diffusion_prior,
                                        int(cfg["ddim_steps"]), float(cfg["ddim_eta"]),
                                        require_grad=True, **ddim_kw)
                lz = torch.nn.functional.mse_loss(v_gen_i, v_patch)
                lz.backward()
                total_lz = total_lz + float(lz.item())
            opt_z.step()
            loss_z_sum = total_lz

        # ── (b) decode v_gen (no grad) ─────────────────────────────────────────
        with torch.no_grad():
            parts = [_ddim_sample(zi, diffusion_prior,
                                   int(cfg["ddim_steps"]), float(cfg["ddim_eta"]),
                                   require_grad=False, **ddim_kw)
                     for zi in z_list]
            v_gen = _patch_blend(parts, positions, overlaps, W)

        # ── (c) v step: wave loss + λ · ||v - v_gen||^2 ──────────────────────
        lam = _lambda(step)
        opt_v.zero_grad(set_to_none=True)
        pred = forward_op(v)
        obs_loss = (pred.float() - seismic_obs.float()).abs().mean()
        if lam > 0:
            reg = torch.nn.functional.mse_loss(v, v_gen.detach())
            loss = obs_loss + lam * reg
            reg_val = float(reg.item())
        else:
            loss = obs_loss
            reg_val = 0.0
        loss.backward()
        opt_v.step()
        if sched_v is not None:
            sched_v.step()
        if cfg["clamp"]:
            with torch.no_grad():
                v.data.clamp_(-1.0, 1.0)

        with torch.no_grad():
            mae = (v - vt_n).abs().mean()
            rmse = ((v - vt_n) ** 2).mean().sqrt()
            ssim_val = ssim(((v + 1) / 2).clamp(0, 1), ((vt_n + 1) / 2).clamp(0, 1))
            mae_g = (v_gen - vt_n).abs().mean()
            rmse_g = ((v_gen - vt_n) ** 2).mean().sqrt()
            ssim_g = ssim(((v_gen + 1) / 2).clamp(0, 1), ((vt_n + 1) / 2).clamp(0, 1))

        history["total_loss"].append(float(loss.item()))
        history["obs_loss"].append(float(obs_loss.item()))
        history["reg_loss"].append(reg_val)
        history["mae"].append(float(mae.cpu()))
        history["rmse"].append(float(rmse.cpu()))
        history["ssim"].append(float(ssim_val.mean().cpu()))
        history["mae_vgen"].append(float(mae_g.cpu()))
        history["rmse_vgen"].append(float(rmse_g.cpu()))
        history["ssim_vgen"].append(float(ssim_g.mean().cpu()))
        history["z_loss"].append(loss_z_sum)

        if progress and (step % log_interval == 0 or step == n_iters - 1):
            pbar.set_postfix({"MAE": history["mae"][-1],
                               "SSIM": history["ssim"][-1],
                               "λ": lam, "z_l": loss_z_sum})

    v_final = v.detach().clamp(-1.0, 1.0)

    with torch.no_grad():
        v_pred_phys = v_denorm(v_final)[0, 0].cpu().numpy()
        v_init_phys = v_denorm(torch.from_numpy(init_n_np)).numpy()
    return InversionResult(
        velocity_pred_phys=v_pred_phys,
        velocity_init_phys=v_init_phys,
        velocity_true_phys=vt[0, 0].cpu().numpy(),
        history=history,
        method="dlo",
        params=cfg,
    )


# =============================================================================
# DiffusionFWI on 70x190 — patched epsilon prediction + arithmetic average merge
# =============================================================================

def _gaussian_kernel_2d(sigma: float, ksize: int, device, dtype) -> torch.Tensor:
    half = (ksize - 1) / 2.0
    coords = torch.arange(ksize, device=device, dtype=dtype) - half
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2 = g.view(-1, 1) * g.view(1, -1)
    return k2.view(1, 1, ksize, ksize)


def _smooth2d(x: torch.Tensor, sigma: float, ksize: int = 5) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = _gaussian_kernel_2d(sigma, ksize, x.device, x.dtype)
    pad = ksize // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k)


def _smooth2d_aniso(x: torch.Tensor, sigma_v: float, sigma_h: float,
                     ksize: int = 5) -> torch.Tensor:
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


def _ddpm_posterior_step(x_t: torch.Tensor, eps_pred: torch.Tensor, t: int,
                          alphas_cumprod: torch.Tensor, *,
                          clip_x0: bool = True,
                          noise_scale: float = 0.001) -> torch.Tensor:
    """Standard DDPM reverse step: (x_t, eps) -> x_{t-1} with optional weak noise.

    Mirrors Wang 2023 / Taufik ilvrefwi ``p_sample_wf`` (noise_scale=0.001).
    """
    abar_t = alphas_cumprod[int(t)].to(x_t.device).float()
    abar_prev = (alphas_cumprod[int(t) - 1].to(x_t.device).float() if t > 0
                  else torch.tensor(1.0, device=x_t.device, dtype=torch.float32))
    x0 = (x_t - (1.0 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
    if clip_x0:
        x0 = x0.clamp(-1.0, 1.0)
    beta_t = 1.0 - abar_t / abar_prev
    coef_x0 = beta_t * abar_prev.sqrt() / (1.0 - abar_t)
    coef_xt = (1.0 - abar_prev) * (1.0 - beta_t).sqrt() / (1.0 - abar_t)
    mean = coef_x0 * x0 + coef_xt * x_t
    if t > 0 and noise_scale != 0.0:
        var = (beta_t * (1.0 - abar_prev) / (1.0 - abar_t)).clamp(min=1e-20)
        return mean + noise_scale * var.sqrt() * torch.randn_like(x_t)
    return mean


def _patched_eps(v: torch.Tensor, prior: DiffusionPrior, t: int,
                  positions: List[Tuple[int, int]]) -> torch.Tensor:
    """Predict eps_theta(v, t) on overlapping patches and merge with simple
    arithmetic averaging in overlap regions (matches ``merge_data_to_size``).

    v: (B, 1, H, W). Each patch is (B, 1, H, P=patch_size). Independent forward
    through the U-Net per patch; the predicted eps tensors are summed back into
    a full-resolution (B, 1, H, W) buffer and divided by a coverage count map.
    """
    B, C, H, W = v.shape
    eps_sum = torch.zeros_like(v)
    count = torch.zeros_like(v)
    t_tensor = torch.full((B,), int(t), device=v.device, dtype=torch.long)
    for (s, e) in positions:
        patch = v[:, :, :, s:e]
        eps_patch = prior.wrapper(patch, t_tensor).sample
        eps_sum[:, :, :, s:e] = eps_sum[:, :, :, s:e] + eps_patch
        count[:, :, :, s:e] = count[:, :, :, s:e] + 1.0
    return eps_sum / count.clamp(min=1.0)


def _fwi_inner_loop_norm(
    v_init: torch.Tensor,
    *,
    forward_op: WaveEqFWIForward,
    target_seismic: torch.Tensor,
    n_steps: int,
    lr: float,
    obs_loss_kind: str,
    optim: str,
    use_scheduler: bool,
    grad_clip: Optional[float],
    grad_normalize: bool,
    grad_smooth_sigma: float,
    grad_smooth_sigma_v: float,
    grad_smooth_kernel: int,
    velocity_blur_kernel: int,
    velocity_blur_sigma: float,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """K-step FWI in normalized ([-1, 1]) velocity domain.

    Matches Taufik ilvrefwi/diffefwi ``fwi_loop`` ordering:
      ① clamp + velocity blur → ② forward+backward → ③ step-0 baseline →
      ④ grad_normalize → ⑤ grad_smooth (refresh baseline) → ⑥ grad_clip →
      ⑦ optimizer.step / scheduler.step.
    """
    v = v_init.detach().clone().requires_grad_(True)
    if optim == "adam":
        optimizer = torch.optim.Adam([{"params": [v], "lr": lr}])
    elif optim == "lbfgs":
        optimizer = torch.optim.LBFGS([v])
    else:
        raise ValueError(f"unknown optim: {optim!r}")
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps + 1, 0)
                  if use_scheduler else None)

    losses: List[float] = []
    grad_baseline: Optional[float] = None

    for step in range(n_steps):
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)
            if velocity_blur_kernel > 0:
                v.data = _smooth2d(v.data, velocity_blur_sigma, velocity_blur_kernel)

        optimizer.zero_grad()
        pred = forward_op(v)
        if obs_loss_kind == "l1":
            diff = (pred.float() - target_seismic.float()).abs()
        elif obs_loss_kind == "l2":
            diff = (pred.float() - target_seismic.float()) ** 2
        else:
            raise ValueError(f"unknown obs_loss: {obs_loss_kind!r}")
        if mask is not None:
            diff = diff * mask
            denom = mask.sum().clamp(min=1.0)
            loss = diff.sum() / denom
        else:
            loss = diff.mean()
        loss.backward()

        if v.grad is not None:
            if step == 0:
                grad_baseline = float(v.grad.abs().max().item())
            if grad_normalize and grad_baseline is not None and grad_baseline > 0:
                v.grad.data.div_(grad_baseline)
            if grad_smooth_sigma > 0:
                if grad_smooth_sigma_v > 0:
                    v.grad.data = _smooth2d_aniso(
                        v.grad.data, grad_smooth_sigma_v, grad_smooth_sigma,
                        grad_smooth_kernel,
                    )
                else:
                    v.grad.data = _smooth2d(
                        v.grad.data, grad_smooth_sigma, grad_smooth_kernel,
                    )
                grad_baseline = float(v.grad.abs().max().item())
            if grad_clip is not None and grad_baseline is not None and grad_baseline > 0:
                torch.nn.utils.clip_grad_norm_([v], max_norm=grad_clip * grad_baseline)

        if optim == "lbfgs":
            optimizer.step(lambda: loss)
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
    return v.detach(), losses


def run_diffusion_fwi_inversion(
    seismic_obs: torch.Tensor,
    velocity_true_phys: torch.Tensor,
    forward_op: WaveEqFWIForward,
    device: torch.device,
    diffusion_prior: DiffusionPrior,
    params: Optional[dict] = None,
    progress: bool = True,
    log_interval: int = 5,
) -> InversionResult:
    """DiffusionFWI (Wang 2023) on Marmousi 70x190 using sliding-window patches.

    At each reverse diffusion step:
      1. Predict eps via the U-Net on overlapping 70x70 patches at positions
         [(0,70), (60,130), (120,190)] independently; merge the per-patch eps
         maps back to (1,1,70,190) by simple arithmetic averaging in overlaps
         (count mask).
      2. DDPM posterior step x_t -> x_{t-1} (μ + 0.001·σ·z by default).
      3. K-step FWI gradient descent on the resulting seed velocity.

    Hyperparameters mirror DIFFUSION_FWI_DEFAULTS (the same set used in the
    OpenFWI scenario).
    """
    if diffusion_prior is None:
        raise ValueError("diffusion-fwi requires diffusion_prior")
    cfg = {**DIFFUSION_FWI_DEFAULTS, **(params or {})}

    torch.manual_seed(int(cfg["seed"]))

    vt = velocity_true_phys.float()
    vt_dev = vt.to(device)
    vt_n = v_norm(vt_dev)

    H, W = vt.shape[2], vt.shape[3]
    P = diffusion_prior.patch_size
    if H != P:
        raise ValueError(f"diffusion-fwi expects height == patch_size ({P}), got {H}")
    positions, _ = _patch_positions(W, P)

    init_n_np = smoothed_initial_norm(vt[0, 0].cpu().numpy(), cfg["sigma_init"])
    v = torch.from_numpy(init_n_np).view(1, 1, H, W).to(device).clamp(-1.0, 1.0)

    target_seismic = seismic_obs.to(device).float()
    diffusion_prior.alphas_cumprod = diffusion_prior.alphas_cumprod.to(device)

    init_t = int(cfg["init_time_step"])
    K = int(cfg["fwi_iters_per_step"])
    noise_scale = float(cfg["ddpm_noise_scale"])
    n_reverse = diffusion_prior.num_train_timesteps - init_t

    history: Dict[str, List[float]] = {
        "total_loss": [], "obs_loss": [], "reg_loss": [],
        "ssim": [], "mae": [], "rmse": [],
    }

    with torch.no_grad():
        mae = (v - vt_n).abs().mean()
        rmse = ((v - vt_n) ** 2).mean().sqrt()
        ssim_v = ssim(((v + 1) / 2).clamp(0, 1), ((vt_n + 1) / 2).clamp(0, 1))
    history["mae"].append(float(mae.cpu()))
    history["rmse"].append(float(rmse.cpu()))
    history["ssim"].append(float(ssim_v.mean().cpu()))
    history["total_loss"].append(0.0); history["obs_loss"].append(0.0); history["reg_loss"].append(0.0)

    pbar = reversed(range(n_reverse))
    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(list(reversed(range(n_reverse))), desc="[diffusion-fwi]", unit="t")

    for t_curr in pbar:
        with torch.no_grad():
            eps_pred = _patched_eps(v, diffusion_prior, int(t_curr), positions)
            v_seed = _ddpm_posterior_step(
                v, eps_pred, int(t_curr), diffusion_prior.alphas_cumprod,
                clip_x0=True, noise_scale=noise_scale,
            ).clamp(-1.0, 1.0)

        v, fwi_losses = _fwi_inner_loop_norm(
            v_seed,
            forward_op=forward_op,
            target_seismic=target_seismic,
            n_steps=K,
            lr=float(cfg["lr"]),
            obs_loss_kind=str(cfg["obs_loss"]),
            optim=str(cfg["optim"]),
            use_scheduler=bool(cfg["use_scheduler"]),
            grad_clip=cfg["grad_clip"],
            grad_normalize=bool(cfg["grad_normalize"]),
            grad_smooth_sigma=float(cfg["grad_smooth_sigma"]),
            grad_smooth_sigma_v=float(cfg["grad_smooth_sigma_v"]),
            grad_smooth_kernel=int(cfg["grad_smooth_kernel"]),
            velocity_blur_kernel=int(cfg["velocity_blur_kernel"]),
            velocity_blur_sigma=float(cfg["velocity_blur_sigma"]),
        )
        v = v.clamp(-1.0, 1.0)
        if cfg["vel_blur_sigma"] > 0:
            v = _smooth2d(v, float(cfg["vel_blur_sigma"]), int(cfg["vel_blur_kernel"]))
            v = v.clamp(-1.0, 1.0)

        with torch.no_grad():
            mae = (v - vt_n).abs().mean()
            rmse = ((v - vt_n) ** 2).mean().sqrt()
            ssim_v = ssim(((v + 1) / 2).clamp(0, 1), ((vt_n + 1) / 2).clamp(0, 1))
        last_obs = fwi_losses[-1] if fwi_losses else 0.0
        history["mae"].append(float(mae.cpu()))
        history["rmse"].append(float(rmse.cpu()))
        history["ssim"].append(float(ssim_v.mean().cpu()))
        history["obs_loss"].append(last_obs)
        history["reg_loss"].append(0.0)
        history["total_loss"].append(last_obs)

        if progress and (t_curr % log_interval == 0 or t_curr == 0):
            pbar.set_postfix({"t": int(t_curr),
                               "MAE": history["mae"][-1],
                               "SSIM": history["ssim"][-1]})

    v_final = v.detach().clamp(-1.0, 1.0)
    with torch.no_grad():
        v_pred_phys = v_denorm(v_final)[0, 0].cpu().numpy()
        v_init_phys = v_denorm(torch.from_numpy(init_n_np)).numpy()
    return InversionResult(
        velocity_pred_phys=v_pred_phys,
        velocity_init_phys=v_init_phys,
        velocity_true_phys=vt[0, 0].cpu().numpy(),
        history=history,
        method="diffusion-fwi",
        params=cfg,
    )
