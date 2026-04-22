#!/usr/bin/env python3
"""
沿 EDM PF-ODE 采样轨迹，计算去噪器 D(x, σ) 关于输入 x 的 Jacobian 最大奇异值（谱范数）。

方法：幂迭代（power iteration）
  - Jv  = ∂(D · v) / ∂x，通过 autograd 反向传播实现
  - J'u = ∂(D(x)) / ∂x |^T · u，同样通过 autograd 实现
  - 交替迭代 v ← J'Jv / ||J'Jv||，收敛到最大奇异向量
  - 对应奇异值 σ_max = ||Jv||

Run（Manifold_constrained_FWI 目录）:
  uv run python exp/jacobian_analysis/demo_jacobian_analysis.py
  uv run python exp/jacobian_analysis/demo_jacobian_analysis.py --config /path/to/custom_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cell.Network import EDMPrecond


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(sc: dict, device: torch.device) -> EDMPrecond:
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
    model_path = _ROOT / sc["train"]["model_path"]
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# ODE trajectory — collect (x_t, sigma_t) at each step
# ---------------------------------------------------------------------------

def collect_trajectory(
    net: EDMPrecond,
    latent: torch.Tensor,         # (1, 1, H, W)
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    solver: str,
) -> tuple[list[torch.Tensor], list[float]]:
    """
    Run PF-ODE and return all intermediate states and sigma values.
    Returns:
        xs     : list of (1,1,H,W) tensors, one per step (including start)
        sigmas : list of float sigma values corresponding to xs
    """
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latent.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    xs = []
    sigmas = []

    x = latent.clone().detach()
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        xs.append(x.detach().clone())
        sigmas.append(float(t_cur.item()))

        with torch.no_grad():
            denoised = net(x, t_cur, None).float()
            d_cur = (x - denoised) / t_cur
            h = t_next - t_cur

            if solver == "heun" and t_next != 0:
                x_prime = x + h * d_cur
                denoised_prime = net(x_prime, t_next, None).float()
                d_prime = (x_prime - denoised_prime) / t_next
                x = x + h * (0.5 * d_cur + 0.5 * d_prime)
            else:
                x = x + h * d_cur

    # Last denoising step (t -> 0 handled by final denoised call)
    xs.append(x.detach().clone())
    sigmas.append(float(t_steps[-2].item()))  # last non-zero sigma

    return xs, sigmas


# ---------------------------------------------------------------------------
# Spectral norm via power iteration
# ---------------------------------------------------------------------------

def spectral_norm_power_iter(
    net: EDMPrecond,
    x: torch.Tensor,
    sigma: float,
    num_iter: int = 30,
    num_samples: int = 3,
) -> float:
    """
    Estimate the spectral norm (largest singular value) of J = ∂D(x,σ)/∂x
    using power iteration on J^T J.

    σ_max(J) = sqrt(λ_max(J^T J))
    Power iteration: v ← J^T J v / ||J^T J v||
    After convergence: σ_max ≈ ||J v|| / ||v||
    """
    def D_fn(x_):
        sigma_t = torch.tensor([sigma], device=x_.device, dtype=torch.float32)
        return net(x_, sigma_t, None).float()

    best = 0.0

    for _ in range(num_samples):
        v = torch.randn_like(x)
        v = v / v.norm()

        for _ in range(num_iter):
            # Jv via forward-mode AD
            _, Jv = torch.autograd.functional.jvp(D_fn, (x.detach(),), (v,), strict=False)
            # J^T(Jv) via reverse-mode AD
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                D = net(x_in, torch.tensor([sigma], device=x.device, dtype=torch.float32), None).float()
                JtJv = torch.autograd.grad(D, x_in, grad_outputs=Jv.detach(), retain_graph=False)[0]
            v = JtJv
            norm = v.norm()
            if norm < 1e-12:
                break
            v = v / norm

        # Spectral norm ≈ ||Jv|| with ||v|| = 1
        _, Jv_final = torch.autograd.functional.jvp(D_fn, (x.detach(),), (v,), strict=False)
        best = max(best, float(Jv_final.norm().item()))

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_jacobian_analysis.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Resolve model path
    sc = cfg["sampler"]
    sc["train"]["model_path"] = str(_ROOT / sc["train"]["model_path"])

    print("Loading model...")
    net = load_model(sc, device)
    net.eval()

    tc = cfg["trajectory"]
    pi = cfg["power_iteration"]
    n_traj = cfg["experiment"]["num_trajectories"]

    # Build sigma schedule (same as ODE sampler)
    num_steps = tc["num_steps"]
    sigma_min = tc["sigma_min"]
    sigma_max = tc["sigma_max"]
    rho = tc["rho"]

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)
    sigma_schedule = [float(t.item()) for t in t_steps]  # decreasing

    print(f"Trajectory: {num_steps} steps, σ: {sigma_schedule[0]:.3f} → {sigma_schedule[-1]:.4f}")
    print(f"Running {n_traj} trajectories × {num_steps} steps × {pi['num_iter']} power iters...")

    all_spec_norms = []  # shape: (n_traj, num_steps)

    for traj_idx in range(n_traj):
        print(f"\n  Trajectory {traj_idx + 1}/{n_traj}")
        # Sample random initial noise
        latent = torch.randn(1, 1, 70, 70, device=device) * sigma_max

        xs, sigmas = collect_trajectory(
            net, latent,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            solver=tc["solver"],
        )
        # xs has num_steps+1 entries; we compute Jacobian at the first num_steps points
        traj_norms = []
        for step_i, (x_t, sig_t) in enumerate(zip(xs[:num_steps], sigmas[:num_steps])):
            spec = spectral_norm_power_iter(
                net, x_t.to(device), sig_t,
                num_iter=pi["num_iter"],
                num_samples=pi["num_samples"],
            )
            traj_norms.append(spec)
            print(f"    step {step_i+1:2d}/{num_steps}  σ={sig_t:.4f}  ||J||={spec:.4f}")

        all_spec_norms.append(traj_norms)

    all_spec_norms = np.array(all_spec_norms)  # (n_traj, num_steps)
    mean_norms = all_spec_norms.mean(axis=0)
    std_norms = all_spec_norms.std(axis=0)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    outdir = _ROOT / cfg["paths"]["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.save(outdir / f"spec_norms_{timestamp}.npy", all_spec_norms)
    np.save(outdir / f"sigma_schedule_{timestamp}.npy", np.array(sigma_schedule[:num_steps]))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    vc = cfg["visualization"]
    fig, axes = plt.subplots(1, 2, figsize=cfg["visualization"]["figsize"])

    sigma_arr = np.array(sigma_schedule[:num_steps])

    for ax, use_log in zip(axes, [False, True]):
        ax.plot(sigma_arr, mean_norms, "b-o", markersize=4, label="mean")
        ax.fill_between(
            sigma_arr,
            mean_norms - std_norms,
            mean_norms + std_norms,
            alpha=0.25,
            color="blue",
            label="±1 std",
        )
        # Individual trajectories (thin)
        for traj_norms in all_spec_norms:
            ax.plot(sigma_arr, traj_norms, "b-", alpha=0.15, linewidth=0.8)

        if use_log:
            ax.set_xscale("log")
            ax.set_title("Spectral Norm of Denoiser Jacobian (log σ)")
        else:
            ax.set_title("Spectral Norm of Denoiser Jacobian")

        ax.set_xlabel("σ (noise level)")
        ax.set_ylabel("||∂D/∂x||₂  (spectral norm)")
        ax.invert_xaxis()  # σ decreases along trajectory
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Max Singular Value of EDM Denoiser Jacobian Along ODE Trajectory\n"
        f"({n_traj} trajectories, {num_steps} steps, {pi['num_iter']} power iters)",
        fontsize=10,
    )
    plt.tight_layout()

    out_fig = outdir / f"jacobian_spectral_norm_{timestamp}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {out_fig}")

    # Summary
    print("\n--- Summary (mean ± std across trajectories) ---")
    print(f"{'σ':>10}  {'||J||₂ mean':>12}  {'std':>8}")
    for sig, m, s in zip(sigma_arr, mean_norms, std_norms):
        print(f"{sig:10.4f}  {m:12.4f}  {s:8.4f}")


if __name__ == "__main__":
    main()
