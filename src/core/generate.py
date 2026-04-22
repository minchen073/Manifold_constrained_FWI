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


# ---------------------------------------------------------------------------
# DDPM / DDIM samplers  (HuggingFace DDPM-scheduler compatible)
# ---------------------------------------------------------------------------


def ddim_build_timesteps(t_start: int, num_steps: int) -> list[int]:
    """Linearly spaced DDIM timesteps from *t_start* down to 0 (num_steps values).

    The spacing is chosen so that ``num_steps=3, t_start=666`` reproduces the
    standard HuggingFace ``set_timesteps(3)`` sequence ``[666, 333, 0]``, while
    a larger ``num_steps`` with the **same** ``t_start`` refines the same ODE
    trajectory without changing what the initial noise ``z`` encodes.

    Examples
    --------
    >>> ddim_build_timesteps(999, 3)   # full-noise, 3-step
    [999, 500, 0]
    >>> ddim_build_timesteps(666, 3)   # default 3-step start (set_timesteps(3))
    [666, 333, 0]
    >>> ddim_build_timesteps(666, 6)   # finer from same t_start
    [666, 533, 400, 266, 133, 0]
    """
    return [
        round(t_start * (num_steps - 1 - i) / max(1, num_steps - 1))
        for i in range(num_steps)
    ]


def _ddim_step(
    model_output: torch.Tensor,
    t: int,
    t_prev: int,
    sample: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    final_alpha_cumprod: float = 1.0,
    eta: float = 0.0,
) -> torch.Tensor:
    """Single DDIM reverse step: x_t → x_{t_prev}.

    Implements the ε-prediction DDIM update with explicit ᾱ look-up::

        pred_x0  = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
        x_{t−1}  = √ᾱ_{t−1} · pred_x0
                 + √(1 − ᾱ_{t−1} − σ²) · ε_θ
                 + σ · ξ,   ξ ~ N(0,I)

    ``clip_sample`` is **not** applied, so gradients flow freely through
    ``pred_x0`` when used inside an optimisation loop.

    Parameters
    ----------
    t_prev < 0 : triggers ``final_alpha_cumprod`` (= 1.0 by default), which
        is the standard DDIM convention for the step "before t = 0".
    """
    device = sample.device
    acp_t    = alphas_cumprod[t].to(device)
    acp_prev = (
        alphas_cumprod[t_prev].to(device)
        if t_prev >= 0
        else torch.tensor(final_alpha_cumprod, dtype=torch.float32, device=device)
    )
    beta_t = 1.0 - acp_t

    # ε-prediction → predicted x_0
    pred_x0 = (sample - beta_t.sqrt() * model_output) / acp_t.sqrt()

    # DDIM variance  σ² = η² · (1−ᾱ_{t−1})/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_{t−1})
    variance = ((1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - acp_t / acp_prev)).clamp(min=0.0)
    sigma    = eta * variance.sqrt()

    coef_dir    = (1.0 - acp_prev - sigma ** 2).clamp(min=0.0).sqrt()
    prev_sample = acp_prev.sqrt() * pred_x0 + coef_dir * model_output

    if eta > 0.0:
        prev_sample = prev_sample + sigma * torch.randn_like(sample)
    return prev_sample


def ddim_sample(
    wrapper,
    alphas_cumprod: torch.Tensor,
    z: torch.Tensor,
    t_start: int,
    num_steps: int,
    eta: float = 0.0,
    final_alpha_cumprod: float = 1.0,
    no_grad: bool = True,
) -> torch.Tensor:
    """DDIM sampling: z (at noise level t_start) → clean velocity field.

    Uses ``ddim_build_timesteps`` to create ``num_steps`` evenly-spaced steps
    from ``t_start`` down to 0.  Because the timestep sequence is derived
    from ``t_start`` (not from ``num_steps`` alone), a higher ``num_steps``
    with the **same** ``t_start`` gives a finer discretisation of the same
    ODE trajectory — the same ``z`` can be reused without distribution shift.

    Parameters
    ----------
    wrapper :
        Callable ``(x, t_tensor) → output`` where ``output.sample`` is the
        predicted noise ε_θ(x_t, t).  Typically an ``OpenFWIUNetWrapper``.
    alphas_cumprod :
        1-D tensor of shape ``(num_train_timesteps,)`` containing ᾱ_t values,
        e.g. ``ddpm_scheduler.alphas_cumprod``.
    z :
        Initial noise, shape ``(B, 1, H, W)``, drawn from N(0, I).
    t_start :
        Effective starting timestep.  Use 999 for "full noise" generation,
        or ``int(ddim_scheduler.timesteps[0])`` to match a specific schedule.
    num_steps :
        Number of denoising steps.
    eta :
        DDIM stochasticity parameter.  0 = deterministic.
    final_alpha_cumprod :
        ᾱ value for the virtual step before t = 0.  Defaults to 1.0
        (``DDIMScheduler`` default when ``set_alpha_to_one=True``).
    no_grad :
        Wrap in ``torch.no_grad()`` for pure inference.
        Set ``False`` when gradients must flow back to ``z``.

    Returns
    -------
    torch.Tensor
        Denoised output, shape ``(B, 1, H, W)``, in the model's output range
        (typically normalised velocity in [-1, 1]).
    """
    timesteps = ddim_build_timesteps(t_start, num_steps)

    def _run(z_in: torch.Tensor) -> torch.Tensor:
        x = z_in
        for i, t in enumerate(timesteps):
            t_prev       = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            model_output = wrapper(x, torch.tensor(t, device=z_in.device)).sample
            x            = _ddim_step(
                model_output, t, t_prev, x,
                alphas_cumprod, final_alpha_cumprod, eta,
            )
        return x

    if no_grad:
        with torch.no_grad():
            return _run(z.detach())
    return _run(z)


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
