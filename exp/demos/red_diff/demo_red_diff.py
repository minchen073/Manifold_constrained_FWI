"""
RED-diff FWI 复现实验（Manifold_constrained_FWI 框架）。

复现 FWI/demos/fwi_then_manifold_init_opt 中的 red_diffeq 方法，移植至本项目的
模型路径、数据路径和正演接口约定。

优化目标：
    min_v  L_wave(v) + λ · R_RED(v)

其中 RED 正则化梯度（手写，不经 autograd）：
    v.grad += λ · (v - D(v + α(t)·ε, max(α(t), σ_min))) · α(t)^(-1) / numel(v)

α(t) = √((1-γ(t))/γ(t))，γ 来自 VP sigmoid β 调度（与 red-diffeq-main 一致）。

运行（Manifold_constrained_FWI 目录）:
    uv run python exp/red_diff/demo_red_diff.py \\
        --config exp/red_diff/config_red_diff.yaml
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
import yaml

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIFOLD_ROOT = Path(__file__).resolve().parents[2]

if str(_MANIFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFOLD_ROOT))

from src.cell.Network import EDMPrecond
from src.core import pytorch_ssim
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
    return seismic_master_forward_modeling(velocity_2d)


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
# VP sigmoid 调度（与 red-diffeq-main 一致）
# ---------------------------------------------------------------------------

def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3.0,
    end: float = 3.0,
    tau: float = 1.0,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, clamp_min, 0.999)


def alphas_cumprod_from_sigmoid_schedule(timesteps: int, **schedule_kwargs) -> torch.Tensor:
    betas = sigmoid_beta_schedule(timesteps, **schedule_kwargs)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).float()


# ---------------------------------------------------------------------------
# RED 引导计算
# ---------------------------------------------------------------------------

def compute_red_guidance(
    model: EDMPrecond,
    x0: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    sigma_min_net: float,
    generator: Optional[torch.Generator] = None,
    *,
    max_timestep: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VP sigmoid γ(t)，VE 加噪 x_kt = x0 + α(t)·ε，α(t) = √((1-γ)/γ)。
    一步去噪 D(x_kt, max(α(t), σ_min))，返回 (guidance, alpha_t, reg_per)。
    guidance = x0 - D；手写梯度：v.grad += λ · guidance · α(t)^(-1) / numel。
    """
    if x0.dim() == 2:
        x0 = x0.unsqueeze(0).unsqueeze(0)
    elif x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    x0 = x0.detach()
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    acp = alphas_cumprod.to(device=device, dtype=dtype)
    T = int(acp.shape[0])
    mt = min(max_timestep if max_timestep is not None else T, T)
    if mt < 1:
        raise ValueError(f"max_timestep must be >= 1, got {mt}")

    t = torch.randint(0, mt, (B,), generator=generator, device=device, dtype=torch.long)
    gamma = acp[t].clamp(1e-7, 1.0 - 1e-7)
    alpha_t = torch.sqrt((1.0 - gamma) / gamma)

    noise = torch.randn(x0.shape, device=device, dtype=dtype, generator=generator)
    x_kt = x0 + alpha_t.view(B, 1, 1, 1) * noise

    sigma_cond = torch.maximum(
        alpha_t, torch.tensor(float(sigma_min_net), device=device, dtype=dtype)
    )
    with torch.no_grad():
        D_pred = model(x_kt, sigma_cond, class_labels=None)

    guidance = x0 - D_pred
    reg_per = (x0 * guidance).view(B, -1).mean(dim=1)
    return guidance, alpha_t, reg_per


# ---------------------------------------------------------------------------
# RED-diff FWI 主循环
# ---------------------------------------------------------------------------

def run_red_diff_fwi(
    v_init: torch.Tensor,
    wavefield_loss_fn: WavefieldLoss,
    obs_wf: torch.Tensor,
    img_res: int,
    edm: EDMPrecond,
    cfg: dict,
    target_batch: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    iters = max(1, int(cfg.get("iterations", 300)))
    lr = float(cfg.get("lr", 0.03))
    reg_lambda = float(cfg.get("reg_lambda", 0.75))
    loss_scale = float(cfg.get("loss_scale", 1.0))
    log_every = max(1, int(cfg.get("log_every", 10)))
    snap_every = max(1, int(cfg.get("snap_every", 30)))
    gif_enabled = bool(cfg.get("gif_enabled", True))
    use_cosine = bool(cfg.get("cosine_annealing", True))
    eta_min = float(cfg.get("eta_min", 0.0))
    noise_seed = int(cfg.get("noise_seed", seed + 17_000))

    Tvp = max(2, int(cfg.get("red_vp_timesteps", 1000)))
    sk = dict(cfg.get("red_sigmoid_schedule_kwargs") or {})
    alphas_cumprod_buf = alphas_cumprod_from_sigmoid_schedule(Tvp, **sk).to(device)
    _ft = cfg.get("red_fixed_timestep")
    red_vp_max_t = int(_ft) if _ft is not None else Tvp
    red_vp_max_t = max(1, min(red_vp_max_t, Tvp))

    sigma_min_net = float(getattr(edm, "sigma_min", 0.002))

    v = torch.nn.Parameter(v_init.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([v], lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=eta_min)
        if use_cosine else None
    )

    loss_hist: list[float] = []
    data_hist: list[float] = []
    mae_hist: list[float] = []
    ssim_hist: list[float] = []
    grad_hist: list[float] = []

    snap_v: list[torch.Tensor] = []
    snap_labels: list[str] = []
    snap_v.append(v_init.detach().cpu().clone())
    snap_labels.append("iter 0 (init)")

    print(
        f"  RED-diff  T={Tvp}  max_t={red_vp_max_t}  reg_lambda={reg_lambda}  "
        f"lr={lr}  iters={iters}",
        flush=True,
    )

    for it in range(iters):
        opt.zero_grad(set_to_none=True)

        # 数据项
        sim = forward_velocity_to_wavefield(v, img_res)
        dl = wavefield_loss_fn(sim, obs_wf) * loss_scale
        dl.backward()

        # RED 引导（手写梯度）
        g_red = torch.Generator(device=device).manual_seed(noise_seed + it)
        guidance, alpha_t, reg_per = compute_red_guidance(
            edm, v, alphas_cumprod_buf, sigma_min_net,
            generator=g_red, max_timestep=red_vp_max_t,
        )
        with torch.no_grad():
            w = (alpha_t ** (-1.0)).view(guidance.shape[0], 1, 1, 1)
            g = guidance * w
            scale = reg_lambda / (v.numel() or 1)
            if v.grad is not None:
                v.grad.add_(g, alpha=scale)

        total = dl.detach() + reg_lambda * reg_per.sum()

        opt.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            v.data.clamp_(-1.0, 1.0)

        gn = v.grad.norm().item() if v.grad is not None else 0.0
        loss_hist.append(float(total.item()))
        data_hist.append(float(dl.detach().item()))
        grad_hist.append(gn)

        with torch.no_grad():
            mae_n, mae_p, ssim_v = velocity_mae_ssim(target_batch, v)
        mae_hist.append(mae_n)
        ssim_hist.append(ssim_v)

        if it % log_every == 0 or it == iters - 1:
            print(
                f"  [RED-diff] {it + 1}/{iters}  total={loss_hist[-1]:.6f}  "
                f"data={data_hist[-1]:.6f}  ||∇v||={gn:.3e}"
                f"\n    {fmt_metrics(mae_n, mae_p, ssim_v)}",
                flush=True,
            )

        if gif_enabled and ((it + 1) % snap_every == 0 or it == iters - 1):
            snap_v.append(v.detach().cpu().clone())
            snap_labels.append(f"iter {it + 1}")

    return dict(
        v=v.detach(),
        loss=loss_hist,
        data_loss=data_hist,
        mae=mae_hist,
        ssim=ssim_hist,
        grad=grad_hist,
        snap_v=snap_v,
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


def _snap_arr(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        return t[0, 0].numpy()
    if t.dim() == 3:
        return t[0].numpy()
    return t.numpy()


def save_comparison(r: dict, true_np: np.ndarray, init_np: np.ndarray, exp_dir: Path) -> None:
    v_np = r["v"][0, 0].cpu().numpy()
    vm = (-1.0, 1.0)
    mn = r["final_metrics"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    panels = [
        (true_np, "True"),
        (init_np, "Init (smooth)"),
        (v_np, f"RED-diff result\nMAE_n={mn[0]:.4f} SSIM={mn[2]:.4f}"),
        (true_np - v_np, "Residual (True - result)"),
    ]
    cmaps = ["viridis", "viridis", "viridis", "RdBu"]
    for ax, (arr, title), cmap in zip(axes, panels, cmaps):
        vmin, vmax = (vm[0], vm[1]) if cmap == "viridis" else (-0.5, 0.5)
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("RED-diff FWI — velocity comparison", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_trajectories(r: dict, exp_dir: Path) -> None:
    iters = list(range(1, len(r["mae"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].plot(iters, r["loss"], color="black", label="total")
    axes[0].plot(iters, r["data_loss"], color="steelblue", linestyle="--", label="data")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, r["mae"], color="steelblue")
    axes[1].set_title("MAE (normalized) vs True")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, r["ssim"], color="darkorange")
    axes[2].set_title("SSIM vs True")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("RED-diff FWI — optimization trajectory", fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "trajectories.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_evolution_panels(r: dict, true_np: np.ndarray, exp_dir: Path) -> None:
    """单行子图展示 v_phys 优化演化过程，第一列为真实速度场参考。"""
    snaps = r.get("snap_v", [])
    labels = r.get("snap_labels", [])
    n = len(snaps)
    if n == 0:
        return

    vm = (-1.0, 1.0)
    ncols = n + 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols, 3.2))
    if ncols == 1:
        axes = [axes]

    im = axes[0].imshow(true_np, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
    axes[0].set_title("True", fontsize=8)
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    for j, snap in enumerate(snaps):
        ax = axes[j + 1]
        arr = _snap_arr(snap)
        im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=vm[0], vmax=vm[1])
        ax.set_title(labels[j] if j < len(labels) else f"snap {j}", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("RED-diff FWI — optimization evolution", fontsize=9)
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

    snaps = r.get("snap_v", [])
    if not snaps:
        return

    rgb = [_vel_to_rgb_uint8(_snap_arr(f)) for f in snaps]
    imageio.mimsave(exp_dir / "evolution.gif", rgb, duration=0.15, loop=0)
    print(f"  GIF: {len(rgb)} frames → evolution.gif", flush=True)


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
        default=str(_SCRIPT_DIR / "config_red_diff.yaml"),
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

    fwi_cfg = cfg.get("red_diff") or {}
    loss_type = str(fwi_cfg.get("loss_type", "mse")).lower().strip()
    w2_cfg = cfg.get("w2_per_trace") or {}
    wavefield_loss_fn = build_wavefield_loss(loss_type, w2_cfg)

    observed = load_observed_wavefield(data_path, file_index, sample_index, device)
    target = load_dataset_sample(data_path, file_index, sample_index, device)
    target_batch = target.unsqueeze(0).unsqueeze(0)

    from scipy.ndimage import gaussian_filter

    t_np = target_batch[0, 0].detach().cpu().numpy()
    v_smooth = (
        torch.from_numpy(gaussian_filter(t_np, sigma=smooth_sigma))
        .float().to(device).unsqueeze(0).unsqueeze(0)
    )

    edm = load_edm(cfg["model"]["model_path"], sampler_cfg, device)

    print(
        f"\n=== RED-diff FWI  |  file={file_index} sample={sample_index}  "
        f"smooth_σ={smooth_sigma} ===\n",
        flush=True,
    )

    results = run_red_diff_fwi(
        v_smooth, wavefield_loss_fn, observed, img_res,
        edm, fwi_cfg, target_batch, device, seed,
    )

    final_metrics = velocity_mae_ssim(target_batch, results["v"].reshape(1, 1, img_res, img_res))
    results["final_metrics"] = final_metrics
    results["v"] = results["v"].reshape(1, 1, img_res, img_res)

    print(f"\n  Final: {fmt_metrics(*final_metrics)}", flush=True)

    # 输出目录
    outdir = str(_SCRIPT_DIR / cfg.get("paths", {}).get("outdir", "demo_output"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(outdir) / f"red_diff_f{file_index}_s{sample_index}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, exp_dir / "config.yaml")

    true_np = t_np
    init_np = v_smooth[0, 0].cpu().numpy()

    save_comparison(results, true_np, init_np, exp_dir)
    save_trajectories(results, exp_dir)
    save_evolution_panels(results, true_np, exp_dir)
    save_evolution_gif(results, exp_dir)

    np.save(exp_dir / "v_result.npy", results["v"][0, 0].cpu().numpy())

    summary = {
        "file_index": file_index,
        "sample_index": sample_index,
        "final_metrics": list(final_metrics),
        "smooth_metrics": list(velocity_mae_ssim(target_batch, v_smooth)),
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "total_loss", "data_loss", "mae_norm", "ssim"])
        for i in range(len(results["mae"])):
            w.writerow([i, results["loss"][i], results["data_loss"][i],
                        results["mae"][i], results["ssim"][i]])

    print(f"\nOutput: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
