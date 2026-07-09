"""指定数据集上跑 N 个测试集样本，计算每方法平均指标 + 柱状图 + 逐样本 log。

用法：
  uv run python Manifold_constrained_FWI/scripts/run_dataset_average.py \\
    --dataset FlatFault-B --n_samples 100

CLI:
  --dataset      数据集名 (FlatVel-B / FlatFault-B / CurveVel-B / CurveFault-B)
  --n_samples    样本数（默认 100）
  --start_idx    起始 sample 索引（默认 0）
  --methods      逗号分隔的方法子集，默认全部 6 个
  --resume       断点续跑：若 log.csv 已存在，跳过已完成样本
  --tag          子目录后缀，默认时间戳

输出：
  exp/dataset_avg/<dataset>/<tag>/
    ├── config.json           运行配置（数据集、方法、超参快照）
    ├── log.csv               每行 = (sample_idx, method, rmse, mae, ssim, time_s)
    ├── summary.json          每方法 mean ± std (RMSE/MAE/SSIM)
    └── bars.png              3 子图柱状（RMSE/MAE/SSIM），6 方法 + error bar

时间预估：6 方法跑完单样本 ~10 min；100 样本 ~16 h/数据集。建议
开 ``--resume`` + 后台跑（``nohup`` / ``screen``），断了能续。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "training"))

_null_fd = os.open(os.devnull, os.O_WRONLY)
_save_fd2 = os.dup(2)
os.dup2(_null_fd, 2)
os.close(_null_fd)
try:
    from openfwi_unet_wrapper import load_openfwi_checkpoint
    from diffusers import DDPMScheduler
finally:
    os.dup2(_save_fd2, 2)
    os.close(_save_fd2)

from src.methods import (
    DiffusionPrior,
    run_tikhonov, run_tv, run_red_diffeq,
    run_diffusion_fwi, run_diffusion_ilvr, run_dlo_fwi,
    apply_missing_traces,
)
from src.seismic import seismic_master_forward_modeling


# =============================================================================
# 数据集配置（与 run_4datasets_compare.py 同源）
# =============================================================================
DATASET_CONFIGS = {
    "FlatVel-B":    {"vel": "model60.npy",   "seis": "data60.npy",
                     "pretrained": "FlatVel-B_DDIM"},
    "FlatFault-B":  {"vel": "vel6_1_35.npy", "seis": "seis6_1_35.npy",
                     "pretrained": "FlatFault-B_DDIM"},
    "CurveVel-B":   {"vel": "model60.npy",   "seis": "data60.npy",
                     "pretrained": "CurveVel-B_DDIM"},
    "CurveFault-B": {"vel": "vel6_1_35.npy", "seis": "seis6_1_35.npy",
                     "pretrained": "CurveFault-B_DDIM"},
}

# =============================================================================
# 方法超参（与 run_4datasets_compare.py 一致；MSE obs loss）
# =============================================================================
METHOD_PARAMS = {
    # 仅 red_diffeq 用 cosine scheduler（与官方一致）；其它物理方法保持恒定 lr 便于历史可比
    # 无正则 FWI（纯波动方程拟合），借用 tikhonov 实现 + reg_lambda=0
    "physical_fwi": {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.0,
                     "smooth_sigma": 10.0, "obs_loss": "l1",
                     "use_scheduler": False},
    "tikhonov":   {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.01,
                   "smooth_sigma": 10.0, "obs_loss": "l1",
                   "use_scheduler": False},
    "tv":         {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.01,
                   "smooth_sigma": 10.0, "obs_loss": "l1",
                   "use_scheduler": False},
    "red_diffeq": {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.75,
                   "smooth_sigma": 10.0, "sigma_x0": 1e-4,
                   "use_time_weight": False, "obs_loss": "l1",
                   "use_scheduler": False},
    # init_time_step 新语义 = "跳过的高噪步数"（官方 ilvrefwi/diffefwi）。900 ⇒ 100 反向步。
    # 稳定化三件套对齐官方 Example-2-efwi.py：grad_norm=True + grad_smooth=1 + gaussian_blur(v,[3,3])。
    #   ① velocity_blur_kernel=3, sigma=0.8 —— torchvision gaussian_blur([3,3]) 隐式 σ
    #   ② grad_smooth_sigma_v=2.0, grad_smooth_sigma=1.0 —— ILVR gaussian_filter(grad, [2, 1])
    #   ③ grad_normalize=True —— 官方 Example-2-efwi.py grad_norm=True
    "diffusion_fwi":  {"init_time_step": 900, "fwi_iters_per_step": 10, "lr": 0.01,
                       "smooth_sigma": 10.0, "obs_loss": "l1",
                       "grad_smooth_sigma":   0,
                       "grad_smooth_sigma_v": 0,
                       "grad_smooth_kernel":  5,
                       "velocity_blur_kernel": 3,
                       "velocity_blur_sigma":  0.1,
                       "grad_normalize":       True},
    # ilvr_factor_schedule 对齐官方 BP-salt：Ns=[32,16,8,4]，按 t_curr 索引（自动重复到反向步数）
    "diffusion_ilvr": {"init_time_step": 900, "fwi_iters_per_step": 10, "lr": 0.03,
                       "smooth_sigma": 10.0, "obs_loss": "l1",
                       "ilvr_factor": 4, "ilvr_factor_schedule": [32, 16, 8, 4],
                       "ilvr_weight": 0.05,
                       "ilvr_ref": "current", "ilvr_domain": "xt",
                       # 稳定化三件套同 diffusion_fwi
                       "grad_smooth_sigma":   1.0,
                       "grad_smooth_sigma_v": 2.0,
                       "grad_smooth_kernel":  5,
                       "velocity_blur_kernel": 3,
                       "velocity_blur_sigma":  0.8,
                       "grad_normalize":       True},
    "dlo_fwi": {"n_iters": 300, "lr_v": 0.03, "lr_z": 0.02, "ddim_steps": 3,
                "lambda_max": 0.5, "warmup_steps": 0, "ramp_steps": 0,
                "phase2_steps": 0,
                "smooth_sigma": 10.0, "obs_loss": "l1",
                "use_scheduler": False},
}

METHOD_FNS = {
    "physical_fwi": run_tikhonov,   # 无正则 FWI（reg_lambda=0）
    "tikhonov": run_tikhonov,
    "tv": run_tv,
    "red_diffeq": run_red_diffeq,
    "diffusion_fwi": run_diffusion_fwi,
    "diffusion_ilvr": run_diffusion_ilvr,
    "dlo_fwi": run_dlo_fwi,
}

DISPLAY_NAMES = {
    "physical_fwi": "Physical FWI",
    "tikhonov": "Tikhonov",
    "tv": "TV",
    "red_diffeq": "RED-DiffEq",
    "diffusion_fwi": "DiffusionFWI",
    "diffusion_ilvr": "DiffusionILVR",
    "dlo_fwi": "DLO (Ours)",
}

PLOT_COLORS = {
    "physical_fwi": "#bcbd22",
    "tikhonov": "#888888",
    "tv": "#1f77b4",
    "red_diffeq": "#2ca02c",
    "diffusion_fwi": "#ff7f0e",
    "diffusion_ilvr": "#d62728",
    "dlo_fwi": "#9467bd",
}

# DEFAULT_METHODS = ["physical_fwi", "tikhonov", "tv","diffusion_fwi", "red_diffeq", "dlo_fwi"]
# DEFAULT_METHODS = ["diffusion_fwi"]
DEFAULT_METHODS = ["red_diffeq"]


# =============================================================================
def load_diffusion_prior(pretrained_subdir: str, device: torch.device) -> DiffusionPrior:
    ckpt = _ROOT / "pretrained_model" / pretrained_subdir / "model.pt"
    wrapper = load_openfwi_checkpoint(ckpt, map_location="cpu").to(device).eval()
    wrapper.requires_grad_(False)
    sched = DDPMScheduler(
        num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    return DiffusionPrior(
        wrapper=wrapper,
        alphas_cumprod=sched.alphas_cumprod.to(device).float(),
        num_train_timesteps=1000,
    )


def add_observation_noise(
    seis: torch.Tensor, sigma: float, noise_type: str, seed: int,
) -> tuple[torch.Tensor, float]:
    """对齐论文 (arXiv 2509.21659v2, Sec.3 result.tex L133-138)：直接对 seismic data 加
    Gaussian(σ) 或 Laplacian(scale=σ) 噪声。返回 (noisy_seis, snr_db)。

    确定性：用 sample-级 seed 起一个独立 generator，方法间共享同一 noisy 观测。
    sigma<=0 → 直接返回原 tensor + inf SNR。
    """
    if sigma <= 0.0:
        return seis, float("inf")
    g = torch.Generator(device=seis.device).manual_seed(int(seed))
    if noise_type == "gaussian":
        eps = torch.randn(seis.shape, generator=g, device=seis.device, dtype=seis.dtype)
    elif noise_type == "laplacian":
        # Laplace(0, 1) via inverse-CDF: U~Uniform(-0.5, 0.5) → -sign(U)*log(1-2|U|)
        u = torch.rand(seis.shape, generator=g, device=seis.device, dtype=seis.dtype) - 0.5
        eps = -torch.sign(u) * torch.log1p(-2.0 * u.abs())
    else:
        raise ValueError(f"unknown noise_type: {noise_type!r} (use gaussian/laplacian)")
    noise = sigma * eps
    sig_p = float((seis.float() ** 2).mean().item())
    noise_p = float((noise.float() ** 2).mean().item())
    snr_db = 10.0 * float(np.log10(sig_p / noise_p)) if noise_p > 0 else float("inf")
    return seis + noise, snr_db


def parse_methods(s: str | None) -> list[str]:
    if s is None or s.strip() == "":
        return list(DEFAULT_METHODS)
    out = [m.strip() for m in s.split(",") if m.strip()]
    for m in out:
        if m not in METHOD_FNS:
            raise SystemExit(f"unknown method: {m}; choose from {list(METHOD_FNS)}")
    return out


def already_done(log_path: Path, methods: list[str]) -> set[tuple[int, str]]:
    """读 log.csv 看哪些 (sample_idx, method) 已完成。"""
    done: set[tuple[int, str]] = set()
    if not log_path.is_file():
        return done
    with open(log_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((int(row["sample_idx"]), row["method"]))
            except (KeyError, ValueError):
                continue
    return done


def append_log(log_path: Path, header: list[str], row: dict) -> None:
    write_header = not log_path.is_file()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Per-dataset 100-sample averaged metrics.")
    parser.add_argument("--dataset", default="CurveVel-B", choices=list(DATASET_CONFIGS))
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--methods", type=str, default=None,
                        help="comma-separated method subset; default = all 6")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing log.csv (skip completed (sample, method) pairs)")
    parser.add_argument("--tag", type=str, default=None,
                        help="subdir tag (default = timestamp). 用同一 tag 配合 --resume 续跑")
    parser.add_argument("--noise_sigma", type=float, default=0,
                        help="加在 seismic data 上的噪声幅度；论文取 0.1/0.2/0.3/0.4/0.5。"
                             "0 表示干净数据（默认）")
    parser.add_argument("--noise_type", type=str, default="gaussian",
                        choices=["gaussian", "laplacian"],
                        help="噪声分布：gaussian (std=σ) 或 laplacian (scale=σ)。对齐论文 Fig. result_openfwi(d)(e)")
    parser.add_argument("--missing_number", type=int, default=10,
                        help="随机置零的 receiver 道数。"
                             "0 表示完整观测（默认）。所有 source 共享同一组缺失索引。")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU。")
    device = torch.device("cuda:0")

    methods = parse_methods(args.methods)
    ds_cfg = DATASET_CONFIGS[args.dataset]
    if args.tag:
        tag = args.tag
    else:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.noise_sigma > 0:
            tag += f"_{args.noise_type[:5]}{args.noise_sigma:.2f}"
        if args.missing_number > 0:
            tag += f"_miss{args.missing_number}"
    out_dir = _ROOT / "exp" / "dataset_avg" / args.dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    # rmse/mae/ssim 来自各方法的 velocity_pred（DLO 为物理速度场 v）。
    header = ["sample_idx", "method", "rmse", "mae", "ssim", "time_s",
              "noise_sigma", "noise_type", "snr_db", "missing_number"]

    # 配置快照（即便 resume 也每次刷新最新一份）
    config = {
        "dataset": args.dataset,
        "vel_file": ds_cfg["vel"],
        "seis_file": ds_cfg["seis"],
        "pretrained": ds_cfg["pretrained"],
        "n_samples": args.n_samples,
        "start_idx": args.start_idx,
        "methods": methods,
        "method_params": {m: METHOD_PARAMS[m] for m in methods},
        "tag": tag,
        "noise_sigma": float(args.noise_sigma),
        "noise_type": args.noise_type if args.noise_sigma > 0 else "none",
        "missing_number": int(args.missing_number),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 数据 + 扩散先验
    vel_arr = np.load(_ROOT / "data" / args.dataset / ds_cfg["vel"], mmap_mode="r")
    seis_arr = np.load(_ROOT / "data" / args.dataset / ds_cfg["seis"], mmap_mode="r")
    n_avail = vel_arr.shape[0]
    end_idx = min(args.start_idx + args.n_samples, n_avail)
    sample_ids = list(range(args.start_idx, end_idx))
    print(f"[avg] dataset={args.dataset}  samples=[{args.start_idx},{end_idx})  "
          f"methods={methods}  out={out_dir}")

    needs_diff = any(m not in ("physical_fwi", "tikhonov", "tv") for m in methods)
    prior = load_diffusion_prior(ds_cfg["pretrained"], device) if needs_diff else None

    common_static = dict(forward_fn=seismic_master_forward_modeling, device=device)

    done = already_done(log_path, methods) if args.resume else set()
    if done:
        print(f"[avg] resume mode: skipping {len(done)} already-completed (sample,method) pairs")

    t_start_all = time.time()
    for i, sidx in enumerate(sample_ids):
        vel_np = np.asarray(vel_arr[sidx, 0], dtype=np.float32)            # (70,70)
        seis_np = np.asarray(seis_arr[sidx], dtype=np.float32)             # (5,1000,70)
        vel_t = torch.from_numpy(vel_np).to(device)
        seis_t = torch.from_numpy(seis_np).to(device)
        # 加噪：sample-级 seed 保证所有方法看到同一份 noisy 观测，且 resume 一致
        noise_seed = sidx * 1009 + (0 if args.noise_type == "gaussian" else 1)
        seis_t, snr_db = add_observation_noise(
            seis_t, args.noise_sigma, args.noise_type, noise_seed,
        )
        # 缺失迹：sample-级 seed (与 noise_seed 不同前缀)，所有方法看到相同的 mask
        # 论文 Sec 2.3.3：specific indices held constant across all source gathers
        # 的语义由 apply_missing_traces 内部沿 R 轴选索引并广播到 (S,T,R) 实现
        trace_mask = None
        if args.missing_number > 0:
            mask_seed = sidx * 1013 + 3
            seis_t, trace_mask = apply_missing_traces(
                seis_t, args.missing_number, seed=mask_seed,
            )
        common = {**common_static, "seismic_obs": seis_t, "velocity_true": vel_t,
                  "trace_mask": trace_mask}

        noise_tag = (f"  noise={args.noise_type}(σ={args.noise_sigma:.2f}) SNR={snr_db:.2f}dB"
                     if args.noise_sigma > 0 else "")
        miss_tag = (f"  missing={args.missing_number}/{seis_np.shape[-1]}"
                    if args.missing_number > 0 else "")
        print(f"\n[avg] [{i+1}/{len(sample_ids)}] sample={sidx}  "
              f"vel range {vel_np.min():.0f}-{vel_np.max():.0f} m/s{noise_tag}{miss_tag}")
        for name in methods:
            if (sidx, name) in done:
                print(f"[avg]   {name:<18} skip (done)")
                continue
            params = METHOD_PARAMS[name]
            diffusion = prior if name not in ("physical_fwi", "tikhonov", "tv") else None
            torch.manual_seed(42)
            t0 = time.time()
            try:
                res = METHOD_FNS[name](diffusion=diffusion, params=params, **common)
            except Exception as e:
                print(f"[avg]   {name:<18} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                # 写入失败行，方便后续过滤；rmse=mae=ssim=NaN
                append_log(log_path, header, {
                    "sample_idx": sidx, "method": name,
                    "rmse": float("nan"), "mae": float("nan"), "ssim": float("nan"),
                    "time_s": time.time() - t0,
                    "noise_sigma": float(args.noise_sigma),
                    "noise_type": args.noise_type if args.noise_sigma > 0 else "none",
                    "snr_db": snr_db,
                    "missing_number": int(args.missing_number),
                })
                continue
            dt = time.time() - t0
            row = {
                "sample_idx": sidx,
                "method": name,
                "rmse": float(res.history["rmse"][-1]),
                "mae":  float(res.history["mae"][-1]),
                "ssim": float(res.history["ssim"][-1]),
                "time_s": float(dt),
                "noise_sigma": float(args.noise_sigma),
                "noise_type": args.noise_type if args.noise_sigma > 0 else "none",
                "snr_db": snr_db,
                "missing_number": int(args.missing_number),
            }
            append_log(log_path, header, row)
            print(f"[avg]   {name:<18} {dt:6.1f}s  "
                  f"RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}  SSIM={row['ssim']:.4f}")

        elapsed = time.time() - t_start_all
        eta = (elapsed / (i + 1)) * (len(sample_ids) - i - 1)
        print(f"[avg]   elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min")

    # 释放扩散模型
    if prior is not None:
        del prior
        torch.cuda.empty_cache()

    # ── 汇总 + 柱状图 ────────────────────────────────────────────────────────
    summarize_and_plot(
        out_dir, methods, args.dataset,
        noise_sigma=float(args.noise_sigma), noise_type=args.noise_type,
        missing_number=int(args.missing_number),
    )


def summarize_and_plot(
    out_dir: Path, methods: list[str], dataset_name: str,
    noise_sigma: float = 0.0, noise_type: str = "gaussian",
    missing_number: int = 0,
) -> None:
    """读 log.csv → 计算 mean/std → 写 summary.json + bars.png."""
    log_path = out_dir / "log.csv"
    if not log_path.is_file():
        print("[summary] log.csv 不存在，跳过")
        return

    by_method: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in ["rmse", "mae", "ssim"]}
        for m in methods
    }
    with open(log_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in by_method:
                continue
            for k in ["rmse", "mae", "ssim"]:
                v = row.get(k, "")
                if v == "" or v is None:
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if not np.isnan(fv):
                    by_method[m][k].append(fv)

    def _stat(arr_list: list[float]) -> dict:
        arr = np.array(arr_list, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
        }

    summary: dict[str, dict] = {}
    for m in methods:
        s = {}
        for k in ["rmse", "mae", "ssim"]:
            s[k] = _stat(by_method[m][k])
        summary[m] = s

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] saved {out_dir / 'summary.json'}")

    # 终端打印
    print(f"\n{'='*84}\n[{dataset_name}] averaged metrics over {summary[methods[0]]['rmse']['n']} samples:\n{'='*84}")
    for k, label in [("rmse", "RMSE"), ("mae", "MAE"), ("ssim", "SSIM")]:
        print(f"\n[{label}]")
        for m in methods:
            s = summary[m][k]
            print(f"  {DISPLAY_NAMES[m]:<22} n={s['n']:3d}  mean={s['mean']:.4f}  std={s['std']:.4f}")
    print(f"{'='*84}")

    # ── 柱状图 ─────────────────────────────────────────────────────────────
    metrics = [("rmse", "RMSE (norm.)", "lower"),
               ("mae",  "MAE (norm.)",  "lower"),
               ("ssim", "SSIM",         "higher")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (key, ylabel, direction) in zip(axes, metrics):
        names = [DISPLAY_NAMES[m] for m in methods]
        means = [summary[m][key]["mean"] for m in methods]
        stds  = [summary[m][key]["std"]  for m in methods]
        colors = [PLOT_COLORS[m] for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black",
                      linewidth=0.5, alpha=0.85)
        for xi, m_, s_ in zip(x, means, stds):
            ax.text(xi, m_ + s_ + (max(means) - min(means)) * 0.02,
                    f"{m_:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ({direction} better)")
        ax.grid(axis="y", alpha=0.3)
    parts = []
    if noise_sigma > 0:
        parts.append(f"{noise_type} noise σ={noise_sigma:.2f}")
    if missing_number > 0:
        parts.append(f"missing {missing_number} traces")
    cond_str = f"  [{' | '.join(parts)}]" if parts else "  [clean]"
    fig.suptitle(f"{dataset_name}: averaged metrics over "
                 f"{summary[methods[0]]['rmse']['n']} samples{cond_str}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_dir / "bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] saved {out_dir / 'bars.png'}")


if __name__ == "__main__":
    main()
