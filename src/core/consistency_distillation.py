"""
Consistency distillation (CD) loss for EDMPrecond-style models.

Follows the logic in openai/consistency_models ``KarrasDenoiser.consistency_losses``:
- Teacher: standard EDM preconditioning (``boundary_condition=False``).
- Student / target: boundary preconditioning (``boundary_condition=True``) so the σ→σ_min
  endpoint matches the consistency-model formulation (at σ=σ_min, D(x)=x).

Teacher trajectory uses a Heun step along the probability-flow ODE in x-space, using the
teacher denoiser D(x,σ) (same as ``consistency_models`` when the denoiser is EDM end-to-end).
"""

from __future__ import annotations

import torch


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def get_weightings_uniform(snrs: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(snrs)


def consistency_distillation_loss(
    x_start: torch.Tensor,
    student: torch.nn.Module,
    target: torch.nn.Module,
    teacher: torch.nn.Module,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_scales: int,
    loss_norm: str = "l2",
    weight_schedule: str = "uniform",
) -> torch.Tensor:
    """
    One batch of L_CD^N style loss. All models must be ``EDMPrecond``-compatible:
    ``forward(x, sigma, class_labels=None, boundary_condition=False)``.

    :param x_start: (B, C, H, W) in [-1, 1], same normalization as EDM training.
    :returns: scalar loss (mean over batch).
    """
    noise = torch.randn_like(x_start)
    dims = x_start.ndim

    def denoise_student(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return student(x, t, boundary_condition=True)

    @torch.no_grad()
    def denoise_target(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return target(x, t, boundary_condition=True)

    @torch.no_grad()
    def denoise_teacher(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return teacher(x, t, boundary_condition=False)

    @torch.no_grad()
    def heun_solver(samples: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor) -> torch.Tensor:
        x = samples
        denoiser = denoise_teacher(x, t)
        d = (x - denoiser) / append_dims(t, dims)
        samples = x + d * append_dims(next_t - t, dims)
        denoiser2 = denoise_teacher(samples, next_t)
        next_d = (samples - denoiser2) / append_dims(next_t, dims)
        samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)
        return samples

    b = x_start.shape[0]
    device = x_start.device
    indices = torch.randint(0, num_scales - 1, (b,), device=device)

    t = (
        sigma_max ** (1 / rho)
        + indices.float() / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t2 = (
        sigma_max ** (1 / rho)
        + (indices.float() + 1) / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    x_t = x_start + noise * append_dims(t, dims)

    dropout_state = torch.get_rng_state()
    distiller = denoise_student(x_t, t)

    x_t2 = heun_solver(x_t, t, t2)

    torch.set_rng_state(dropout_state)
    with torch.no_grad():
        distiller_target = denoise_target(x_t2, t2).detach()

    snrs = t**-2
    if weight_schedule == "uniform":
        weights = get_weightings_uniform(snrs)
    else:
        weights = torch.ones_like(snrs)

    if loss_norm == "l2":
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs) * weights
    elif loss_norm == "l1":
        diffs = torch.abs(distiller - distiller_target)
        loss = mean_flat(diffs) * weights
    else:
        raise ValueError(f"loss_norm must be 'l2' or 'l1', got {loss_norm}")

    return loss.mean()


def update_target_ema(target: torch.nn.Module, source: torch.nn.Module, rate: float) -> None:
    """target = rate * target + (1 - rate) * source (same as consistency_models ``update_ema``)."""
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.mul_(rate).add_(ps.data, alpha=1.0 - rate)
