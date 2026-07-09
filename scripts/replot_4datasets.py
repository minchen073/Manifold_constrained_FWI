"""从已有 results.npz 重新绘图，可选对个别数据集换 idx 重跑。

用法
----
纯重画（不重跑，直接读 results.npz）：
    python scripts/replot_4datasets.py exp/4ds_compare/20260518_170254

指定某几个数据集换样本重跑（未指定的从 npz 读）：
    python scripts/replot_4datasets.py exp/4ds_compare/20260518_170254 \\
        --override CurveVel-B:42 --override FlatFault-B:7

输出写到一个新时间戳目录 exp/4ds_compare/<ts>_replot/（除非 --out-dir 指定）。
"""
from __future__ import annotations

import argparse
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

# 从主脚本复用所有配置 + 绘图函数 + load_diffusion_prior
from run_4datasets_compare import (
    DATASETS, METHOD_PARAMS, METHOD_FNS, METHOD_ORDER, DISPLAY_NAMES,
    plot_velocity_compare, plot_convergence_curves, load_diffusion_prior,
)
from src.seismic import seismic_master_forward_modeling


# 与主脚本 npz 里出现的 history 键保持一致
HISTORY_KEYS_BASE = ["rmse", "mae", "ssim", "obs_loss", "reg_loss", "total_loss"]
HISTORY_KEYS_EXTRA = {  # 个别方法多写了几条
    "dlo_phase1": ["rmse_vgen", "mae_vgen", "ssim_vgen"],
}


def load_results_npz(npz_path: Path) -> tuple[dict, dict]:
    """读 results.npz → (all_results, all_inputs)，结构与 run 脚本一致。"""
    d = np.load(npz_path)
    all_results: dict[str, dict[str, dict]] = {}
    all_inputs: dict[str, dict] = {}
    for ds in DATASETS:
        ds_name = ds["name"]
        all_inputs[ds_name] = {
            "vel_np": d[f"{ds_name}__ground_truth"],
            "init_phys": d[f"{ds_name}__init_phys"],
            "seis_np": None,  # 重画无需 seismic
        }
        ds_res: dict[str, dict] = {}
        for name in METHOD_ORDER:
            hist_keys = HISTORY_KEYS_BASE + HISTORY_KEYS_EXTRA.get(name, [])
            history = {}
            for k in hist_keys:
                full = f"{ds_name}__{name}__hist_{k}"
                if full in d:
                    history[k] = np.asarray(d[full])
            ds_res[name] = {
                "velocity_pred": d[f"{ds_name}__{name}__pred"],
                "history": history,
                "time_s": float(d[f"{ds_name}__{name}__time_s"]),
            }
        all_results[ds_name] = ds_res
    return all_results, all_inputs


def rerun_dataset(ds_cfg: dict, new_idx: int, device: torch.device) -> tuple[dict, dict]:
    """对单个数据集换 idx 跑全部 METHOD_ORDER 方法，返回 (ds_results, ds_inputs)。"""
    ds_name = ds_cfg["name"]
    print(f"\n{'='*72}\n[replot-rerun] {ds_name}  idx {ds_cfg['idx']} → {new_idx}\n{'='*72}")

    vel_np = np.load(_ROOT / "data" / ds_name / ds_cfg["vel"])[new_idx, 0].astype(np.float32)
    seis_np = np.load(_ROOT / "data" / ds_name / ds_cfg["seis"])[new_idx].astype(np.float32)
    vel_t = torch.from_numpy(vel_np).to(device)
    seis_t = torch.from_numpy(seis_np).to(device)
    print(f"[replot-rerun]   vel range {vel_np.min():.0f}-{vel_np.max():.0f} m/s")

    prior = load_diffusion_prior(ds_cfg["pretrained"], device)
    common = dict(seismic_obs=seis_t, velocity_true=vel_t,
                  forward_fn=seismic_master_forward_modeling, device=device)

    ds_results: dict[str, dict] = {}
    init_phys: np.ndarray | None = None
    for name in METHOD_ORDER:
        params = METHOD_PARAMS[name]
        diffusion = prior if name not in ("physical_fwi", "tikhonov", "tv") else None
        torch.manual_seed(42)
        t0 = time.time()
        res = METHOD_FNS[name](diffusion=diffusion, params=params, **common)
        dt = time.time() - t0
        ds_results[name] = {
            "velocity_pred": res.velocity_pred,
            "history": res.history,
            "time_s": dt,
        }
        if init_phys is None:
            init_phys = res.velocity_init
        print(f"[replot-rerun]   {name:<18} done {dt:6.1f}s  "
              f"RMSE {res.history['rmse'][0]:.4f}→{res.history['rmse'][-1]:.4f}  "
              f"SSIM {res.history['ssim'][0]:.4f}→{res.history['ssim'][-1]:.4f}")

    del prior
    torch.cuda.empty_cache()

    return ds_results, {"vel_np": vel_np, "init_phys": init_phys, "seis_np": seis_np}


def save_results_npz(all_results: dict, all_inputs: dict, out_path: Path) -> None:
    save: dict[str, np.ndarray] = {}
    for ds_name, ds_res in all_results.items():
        save[f"{ds_name}__ground_truth"] = all_inputs[ds_name]["vel_np"]
        save[f"{ds_name}__init_phys"] = all_inputs[ds_name]["init_phys"]
        for name in METHOD_ORDER:
            r = ds_res[name]
            save[f"{ds_name}__{name}__pred"] = r["velocity_pred"]
            for k, v in r["history"].items():
                save[f"{ds_name}__{name}__hist_{k}"] = np.asarray(v, dtype=np.float32)
            save[f"{ds_name}__{name}__time_s"] = np.float32(r["time_s"])
    np.savez(out_path, **save)


def parse_overrides(items: list[str]) -> dict[str, int]:
    """形如 ['CurveVel-B:42', 'FlatFault-B:7'] → {'CurveVel-B': 42, ...}。"""
    out: dict[str, int] = {}
    valid = {ds["name"] for ds in DATASETS}
    for s in items:
        if ":" not in s:
            raise SystemExit(f"--override 格式应为 DATASET:IDX，得到 {s!r}")
        name, idx = s.split(":", 1)
        if name not in valid:
            raise SystemExit(f"未知数据集 {name!r}，可选: {sorted(valid)}")
        out[name] = int(idx)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path,
                   help="包含 results.npz 的目录，例如 exp/4ds_compare/20260518_170254")
    p.add_argument("--override", action="append", default=[],
                   metavar="DATASET:IDX",
                   help="对指定数据集换 idx 重跑（可多次传）。其余从 npz 读。")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="输出目录。默认 exp/4ds_compare/<ts>_replot/")
    args = p.parse_args()

    npz_path = args.results_dir / "results.npz"
    if not npz_path.exists():
        raise SystemExit(f"找不到 {npz_path}")
    overrides = parse_overrides(args.override)

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _ROOT / "exp" / "4ds_compare" / f"{ts}_replot"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[replot] source = {npz_path}")
    print(f"[replot] out_dir = {out_dir}")
    if overrides:
        print(f"[replot] overrides = {overrides}")

    # 1) 先从 npz 读全部
    all_results, all_inputs = load_results_npz(npz_path)

    # 2) 对 override 的数据集重跑
    if overrides:
        if not torch.cuda.is_available():
            raise SystemExit("重跑需要 CUDA GPU。")
        device = torch.device("cuda:0")
        # 用 DATASETS 模板（拿到 vel/seis/pretrained 字段）
        ds_by_name = {ds["name"]: ds for ds in DATASETS}
        for ds_name, new_idx in overrides.items():
            ds_cfg = dict(ds_by_name[ds_name])
            ds_cfg["idx"] = new_idx  # 仅作日志展示
            ds_results, ds_inputs = rerun_dataset(ds_by_name[ds_name], new_idx, device)
            all_results[ds_name] = ds_results
            all_inputs[ds_name] = ds_inputs

    # 3) 保存合并后的 npz（即使没重跑也存一份，方便链式 replot）
    save_results_npz(all_results, all_inputs, out_dir / "results.npz")
    print(f"[replot] saved: {out_dir / 'results.npz'}")

    # 4) 绘图
    plot_velocity_compare(DATASETS, all_results, all_inputs,
                          out_dir / "velocity_compare.png")
    print(f"[replot] saved: {out_dir / 'velocity_compare.png'}")
    plot_convergence_curves(DATASETS, all_results, out_dir / "curves.png")
    print(f"[replot] saved: {out_dir / 'curves.png'}")


if __name__ == "__main__":
    main()
