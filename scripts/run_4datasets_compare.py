"""4 个 OpenFWI 数据集 × 6 方法的单样本反演对比 (Experiment 1)。

每个数据集挑一个测试集样本，跑 6 个方法（Physical FWI / Tikhonov / TV /
RED-DiffEq / DiffusionFWI / DLO-Phase1），输出：

  exp/4ds_compare/<timestamp>/
    ├── results.npz                 每方法每数据集的 velocity_pred + history
    ├── velocity_compare.png        4×8 grid（GT, Init, 6 methods）
    └── curves.png                  4×3 grid（每行 RMSE/MAE/SSIM，仅画 step
                                    语义对齐的方法）

DLO-Phase1：复用 run_dlo_fwi 接口，``phase2_steps=0`` 仅跑 300 步的 phase1，
直接输出物理速度 v 作反演结果（不进 phase2 的 latent decode）。
DiffusionFWI：init_t=900（100 反向步）× K=10 = 1000 FWI 梯度步，
其 step 语义与物理迭代不同，不参与曲线对比，只出现在最终速度场图中。
"""
from __future__ import annotations

import os
import sys
import time
import warnings
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

# diffusers 导入静默
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
    run_diffusion_fwi, run_dlo_fwi,
)
from src.seismic import seismic_master_forward_modeling


# =============================================================================
# 数据集 / 预训练模型 / 测试样本配置
# =============================================================================
DATASETS = [
    {"name": "FlatVel-B",    "vel": "model60.npy",      "seis": "data60.npy",       "idx": 72,
     "pretrained": "FlatVel-B_DDIM"},
    {"name": "FlatFault-B",  "vel": "vel6_1_35.npy",    "seis": "seis6_1_35.npy",   "idx": 0,
     "pretrained": "FlatFault-B_DDIM"},
    {"name": "CurveVel-B",   "vel": "model60.npy",      "seis": "data60.npy",       "idx": 19,
     "pretrained": "CurveVel-B_DDIM"},
    {"name": "CurveFault-B", "vel": "vel6_1_35.npy",    "seis": "seis6_1_35.npy",   "idx": 21,
     "pretrained": "CurveFault-B_DDIM"},
]
# 72 54 58 17
VEL_VMIN_M_S, VEL_VMAX_M_S = 1500.0, 4500.0


# =============================================================================
# 方法超参（与 run_dataset_average.py 对齐）
# =============================================================================
METHOD_PARAMS = {
    # 无正则 FWI（纯波动方程拟合），借用 tikhonov 实现 + reg_lambda=0
    # 仅 red_diffeq 用 cosine scheduler（与官方一致）；其它物理方法保持恒定 lr 便于历史可比
    "physical_fwi": {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.0,
                     "smooth_sigma": 10.0, "obs_loss": "l1",
                     "use_scheduler": False},
    "tikhonov":   {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.01, "smooth_sigma": 10.0,
                   "obs_loss": "l1",
                   "use_scheduler": False},
    "tv":         {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.01, "smooth_sigma": 10.0,
                   "obs_loss": "l1",
                   "use_scheduler": False},
    "red_diffeq": {"n_iters": 300, "lr": 0.03, "reg_lambda": 0.75, "smooth_sigma": 10.0,
                   "sigma_x0": 1e-4, "use_time_weight": False, "obs_loss": "l1",
                   "use_scheduler": True},
    "diffusion_fwi":  {"init_time_step": 900, "fwi_iters_per_step": 10, "lr": 0.01,
                       "smooth_sigma": 10.0, "obs_loss": "l1",
                       "grad_smooth_sigma":   0,
                       "grad_smooth_sigma_v": 0,
                       "grad_smooth_kernel":  5,
                       "velocity_blur_kernel": 3,
                       "velocity_blur_sigma":  0.2,
                       "grad_normalize":       False},
    # DLO-Phase1：phase2_steps=0 ⇒ 只跑 phase1 300 步，输出物理速度 v 作反演结果
    # （等同 run_dataset_average.py 的 dlo_phase1）
    "dlo_phase1": {"n_iters": 300, "lr_v": 0.03, "lr_z": 0.02, "ddim_steps": 3,
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
    "dlo_phase1": run_dlo_fwi,   # 复用 DLO-FWI 实现，phase2_steps=0 → 仅 phase1
}

DISPLAY_NAMES = {
    "physical_fwi": "Physical FWI",
    "tikhonov": "Tikhonov",
    "tv": "TV",
    "red_diffeq": "RED-DiffEq",
    "diffusion_fwi": "DiffusionFWI",
    "dlo_phase1": "DLO",
}

PLOT_COLORS = {
    "physical_fwi": "#bcbd22",
    "tikhonov": "#888888",
    "tv": "#1f77b4",
    "red_diffeq": "#2ca02c",
    "diffusion_fwi": "#ff7f0e",
    "dlo_phase1": "#8c564b",
}

METHOD_ORDER = ["physical_fwi", "tikhonov", "tv",
                "diffusion_fwi", "red_diffeq", "dlo_phase1"]
# 收敛曲线只画这些（step 语义对齐的；diffusion_fwi step 不对齐，不画）：
CURVE_METHODS = ["physical_fwi", "tikhonov", "tv", "red_diffeq", "dlo_phase1"]


# =============================================================================
# 共享绘图函数（被本脚本和 replot_4datasets.py 复用）
# =============================================================================
def plot_velocity_compare(datasets: list, all_results: dict, all_inputs: dict,
                          out_path: Path) -> None:
    """4 行 × 8 列 速度场对比 grid（GT / Initial / 6 methods）。

    datasets: 形如 [{"name": ds_name}, ...]，只需要 name 字段。
    all_results[ds_name][method_name]["velocity_pred"]：np.ndarray (H, W)
    all_inputs[ds_name]: {"vel_np":..., "init_phys":...}
    """
    n_rows = len(datasets)
    cols_velocity = ["Ground Truth", "Initial"] + [DISPLAY_NAMES[n] for n in METHOD_ORDER]
    n_cols = len(cols_velocity)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows + 0.6))
    if n_rows == 1:
        axes = axes[None, :]

    im = None
    for r, ds in enumerate(datasets):
        ds_name = ds["name"]
        ds_res = all_results[ds_name]
        gt = all_inputs[ds_name]["vel_np"]
        init = all_inputs[ds_name]["init_phys"]
        panels: list[tuple[str, np.ndarray]] = [("Ground Truth", gt), ("Initial", init)]
        for name in METHOD_ORDER:
            panels.append((DISPLAY_NAMES[name], ds_res[name]["velocity_pred"]))

        for c, (title, arr) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(arr, cmap="jet", vmin=VEL_VMIN_M_S, vmax=VEL_VMAX_M_S, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(ds_name, fontsize=12, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.94, top=0.96, bottom=0.04,
                        wspace=0.05, hspace=0.10)
    cax = fig.add_axes([0.955, 0.08, 0.010, 0.84])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("velocity (m/s)", fontsize=11)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_convergence_curves(datasets: list, all_results: dict, out_path: Path) -> None:
    """4 行 × 3 列 (RMSE / MAE / SSIM) 收敛曲线，仅 CURVE_METHODS。"""
    metrics = [("rmse", "RMSE (norm.)", "lower"),
               ("mae",  "MAE (norm.)",  "lower"),
               ("ssim", "SSIM",         "higher")]
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for r, ds in enumerate(datasets):
        ds_name = ds["name"]
        ds_res = all_results[ds_name]
        for c, (key, ylabel, direction) in enumerate(metrics):
            ax = axes[r, c]
            for name in CURVE_METHODS:
                h = ds_res[name]["history"][key]
                ax.plot(np.arange(len(h)), h,
                        label=DISPLAY_NAMES[name],
                        color=PLOT_COLORS[name], linewidth=1.4)
            ax.set_xlabel("iteration step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(f"{ylabel}  ({direction} better)")
            if c == 0:
                ax.text(-0.18, 0.5, ds_name, transform=ax.transAxes,
                        ha="right", va="center", fontsize=12, fontweight="bold",
                        rotation=90)
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        f"Convergence curves (4 datasets × 3 metrics, {len(CURVE_METHODS)} step-aligned methods)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU。")
    device = torch.device("cuda:0")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "exp" / "4ds_compare" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] out_dir = {out_dir}")

    # ── 跑全部数据集 × 全部方法 ──────────────────────────────────────────────
    all_results: dict[str, dict[str, dict]] = {}     # ds_name -> method -> result
    all_inputs: dict[str, dict] = {}                  # ds_name -> {vel_np, seis_np, init_phys}

    for ds in DATASETS:
        ds_name = ds["name"]
        print(f"\n{'='*72}\n[run] {ds_name}\n{'='*72}")

        # 数据
        vel_np = np.load(_ROOT / "data" / ds_name / ds["vel"])[ds["idx"], 0].astype(np.float32)
        seis_np = np.load(_ROOT / "data" / ds_name / ds["seis"])[ds["idx"]].astype(np.float32)
        vel_t = torch.from_numpy(vel_np).to(device)
        seis_t = torch.from_numpy(seis_np).to(device)
        print(f"[run]   data: {ds['vel']}[{ds['idx']}]  vel range "
              f"{vel_np.min():.0f}-{vel_np.max():.0f} m/s")

        # 该数据集的扩散先验
        prior = load_diffusion_prior(ds["pretrained"], device)

        common = dict(seismic_obs=seis_t, velocity_true=vel_t,
                      forward_fn=seismic_master_forward_modeling, device=device)

        ds_results: dict[str, dict] = {}
        for name in METHOD_ORDER:
            params = METHOD_PARAMS[name]
            diffusion = prior if name not in ("tikhonov", "tv") else None
            torch.manual_seed(42)
            t0 = time.time()
            res = METHOD_FNS[name](diffusion=diffusion, params=params, **common)
            dt = time.time() - t0
            ds_results[name] = {
                "velocity_pred": res.velocity_pred,
                "velocity_init": res.velocity_init,
                "history": res.history,
                "time_s": dt,
            }
            print(f"[run]   {name:<18} done {dt:6.1f}s  "
                  f"steps={len(res.history['rmse'])}  "
                  f"RMSE {res.history['rmse'][0]:.4f}→{res.history['rmse'][-1]:.4f}  "
                  f"MAE {res.history['mae'][0]:.4f}→{res.history['mae'][-1]:.4f}  "
                  f"SSIM {res.history['ssim'][0]:.4f}→{res.history['ssim'][-1]:.4f}")

        all_results[ds_name] = ds_results
        all_inputs[ds_name] = {
            "vel_np": vel_np,
            "seis_np": seis_np,
            "init_phys": ds_results["tikhonov"]["velocity_init"],
        }

        # 释放该数据集的扩散模型 GPU 显存
        del prior
        torch.cuda.empty_cache()

    # ── 保存 npz ──────────────────────────────────────────────────────────────
    save: dict[str, np.ndarray] = {}
    for ds_name, ds_res in all_results.items():
        prefix = ds_name
        save[f"{prefix}__ground_truth"] = all_inputs[ds_name]["vel_np"]
        save[f"{prefix}__init_phys"] = all_inputs[ds_name]["init_phys"]
        for name in METHOD_ORDER:
            r = ds_res[name]
            save[f"{prefix}__{name}__pred"] = r["velocity_pred"]
            for k, v in r["history"].items():
                save[f"{prefix}__{name}__hist_{k}"] = np.asarray(v, dtype=np.float32)
            save[f"{prefix}__{name}__time_s"] = np.float32(r["time_s"])
    np.savez(out_dir / "results.npz", **save)
    print(f"\n[run] saved: {out_dir / 'results.npz'}")

    # =========================================================================
    # 图 1：速度场对比 grid，4 行 × 8 列
    # =========================================================================
    plot_velocity_compare(DATASETS, all_results, all_inputs,
                          out_dir / "velocity_compare.png")
    print(f"[run] saved: {out_dir / 'velocity_compare.png'}")

    # =========================================================================
    # 图 2：收敛曲线 4 行 × 3 列（仅 CURVE_METHODS，step 语义对齐）
    # =========================================================================
    plot_convergence_curves(DATASETS, all_results, out_dir / "curves.png")
    print(f"[run] saved: {out_dir / 'curves.png'}")

    # =========================================================================
    # 终态指标汇总（每数据集一行，方法列；终态 = history 最后一格）
    # =========================================================================
    cols = list(METHOD_ORDER)
    col_display = {n: DISPLAY_NAMES[n] for n in cols}

    def get_metric(ds_name: str, col: str, key: str) -> float:
        return all_results[ds_name][col]["history"][key][-1]

    print(f"\n{'='*120}\nFinal metrics (norm. domain):\n{'='*120}")
    for metric_key, label in [("rmse", "RMSE"), ("mae", "MAE"), ("ssim", "SSIM")]:
        print(f"\n[{label}]")
        header = f"{'dataset':<14}" + " ".join(f"{col_display[c]:>15}" for c in cols)
        print(header)
        for ds in DATASETS:
            row = f"{ds['name']:<14}"
            for c in cols:
                row += f" {get_metric(ds['name'], c, metric_key):>15.4f}"
            print(row)
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
