# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""EDM probability-flow ODE samplers (subset of the original generate utilities)."""

import torch

# ----------------------------------------------------------------------------


def edm_sampler_ode(
    net,
    latents,
    class_labels=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=20,
    rho=7,
    alpha=1,
    solver="euler",
):
    """
    PF-ODE from ``sigma_max`` down to ``sigma_min`` (power-law nodes in ``sigma ** (1/rho)``), then 0.
    For denoising a noisy sample ``x_t`` consistent with noise level ``sigma_t``, set
    ``sigma_max=sigma_t``, ``latents=x_t``, and choose ``num_steps`` for the interval
    ``[sigma_max, sigma_min]`` (shorter intervals need fewer steps than full ``sigma_max=80`` runs).
    """
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float32)
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_cur = x_next
        h = t_next - t_cur
        denoised = net(x_cur, t_cur, class_labels).to(torch.float32)
        d_cur = (x_cur - denoised) / t_cur

        if solver == "heun" and t_next != 0:
            x_prime = x_cur + alpha * h * d_cur
            t_prime = t_cur + alpha * h
            denoised_prime = net(x_prime, t_prime, class_labels).to(torch.float32)
            d_prime = (x_prime - denoised_prime) / t_prime
            x_next = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
        else:
            x_next = x_cur + h * d_cur
    return x_next


class EDM_SingleStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_cur, t_cur, t_next, net, class_labels, solver="heun", alpha=1):
        """One PF-ODE step matching ``edm_sampler_ode``."""
        ctx.t_cur = t_cur
        ctx.t_next = t_next
        ctx.net = net
        ctx.class_labels = class_labels
        ctx.solver = solver
        ctx.alpha = alpha
        ctx.h = t_next - t_cur

        ctx.save_for_backward(x_cur)

        h = t_next - t_cur
        denoised = net(x_cur, t_cur, class_labels).to(torch.float32)
        d_cur = (x_cur - denoised) / t_cur

        if solver == "heun" and t_next != 0:
            x_prime = x_cur + alpha * h * d_cur
            t_prime = t_cur + alpha * h
            denoised_prime = net(x_prime, t_prime, class_labels).to(torch.float32)
            d_prime = (x_prime - denoised_prime) / t_prime
            x_next = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
        else:
            x_next = x_cur + h * d_cur

        return x_next

    @staticmethod
    def backward(ctx, grad_output):
        x_cur, = ctx.saved_tensors
        t_cur = ctx.t_cur
        h = ctx.h
        net = ctx.net
        class_labels = ctx.class_labels
        solver = ctx.solver
        alpha = ctx.alpha

        if not x_cur.requires_grad:
            x_cur = x_cur.requires_grad_(True)

        with torch.enable_grad():
            denoised = net(x_cur, t_cur, class_labels).to(torch.float32)

            if solver == "heun" and ctx.t_next != 0:
                d_cur = (x_cur - denoised) / t_cur
                x_prime = x_cur + alpha * h * d_cur
                t_prime = t_cur + alpha * h
                denoised_prime = net(x_prime, t_prime, class_labels).to(torch.float32)
                d_prime = (x_prime - denoised_prime) / t_prime
                x_next = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

                coef1 = h * (1 - 1 / (2 * alpha)) / t_cur
                coef2 = h * (1 / (2 * alpha)) / t_prime
                grad_denoised = torch.autograd.grad(
                    outputs=denoised,
                    inputs=x_cur,
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                grad_xp_xcur = grad_output + alpha * h * (grad_output - grad_denoised) / t_cur
                grad_denoised_prime = torch.autograd.grad(
                    outputs=denoised_prime,
                    inputs=x_cur,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                grad_x_cur = (
                    grad_output
                    + (grad_output - grad_denoised) * coef1
                    + coef2 * (grad_xp_xcur - grad_denoised_prime)
                )

            else:
                grad_denoised = torch.autograd.grad(
                    outputs=denoised,
                    inputs=x_cur,
                    grad_outputs=grad_output * h / t_cur,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                grad_x_cur = grad_output * (1 + h / t_cur) - grad_denoised

        return grad_x_cur, None, None, None, None, None, None


def edm_sampler_ode_latentgrad(
    net,
    latents,
    class_labels=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=20,
    rho=7,
    alpha=1,
    solver="euler",
):
    """Same schedule as ``edm_sampler_ode`` but uses ``EDM_SingleStep`` so latents can receive gradients."""
    batch_size = latents.shape[0]
    if class_labels is not None and class_labels.shape[0] != batch_size:
        raise ValueError(
            f"class_labels batch size ({class_labels.shape[0]}) does not match latents ({batch_size})"
        )

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x = latents
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x = EDM_SingleStep.apply(x, t_cur, t_next, net, class_labels, solver, alpha)
        if i < num_steps - 1:
            torch.cuda.empty_cache()

    return x


def forward_edm_sampler(
    net,
    data,
    class_labels=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=20,
    rho=7,
    alpha=1,
    solver="heun",
):
    """
    Forward (noise-increasing) PF-ODE: integrate from ``sigma_min`` to ``sigma_max``.

    Returns the state at ``sigma_max`` (e.g. noisy velocity field).
    """
    net.eval()
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=data.device)
    t_steps = (
        sigma_min ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)

    x_next = data.to(torch.float32)

    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x_cur = x_next
            h = t_next - t_cur
            denoised = net(x_cur, t_cur, class_labels).to(torch.float32)
            d_cur = (x_cur - denoised) / t_cur

            if solver == "euler":
                x_next = x_cur + h * d_cur
            elif solver == "heun":
                x_prime = x_cur + alpha * h * d_cur
                t_prime = t_cur + alpha * h
                denoised_prime = net(x_prime, t_prime, class_labels).to(torch.float32)
                d_prime = (x_prime - denoised_prime) / t_prime
                x_next = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
            else:
                raise ValueError(f"Unknown solver {solver!r}; use 'euler' or 'heun'.")

            torch.cuda.empty_cache()

    return x_next
