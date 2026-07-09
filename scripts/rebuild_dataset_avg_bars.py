"""Rebuild dataset_avg bar charts per scenario.

Reads scenario directories from `exp/dataset_avg/exp.txt` (4 datasets × 7
scenarios = 28 entries). For each scenario:

  * loads the top-level ``summary.json`` for: physical_fwi, tikhonov, tv,
    red_diffeq, dlo_phase1;
  * **replaces** the bad diffusion_fwi entry with the rerun stored in the
    nested sub-directory's ``summary.json``;
  * drops diffusion_ilvr entirely;
  * renames dlo_phase1 → DLO in display.

Writes the new figure to ``<scenario_dir>/bars_v2.png`` (originals untouched).

Usage:
    uv run python Manifold_constrained_FWI/scripts/rebuild_dataset_avg_bars.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_TXT = REPO_ROOT / "Manifold_constrained_FWI" / "exp" / "dataset_avg" / "exp.txt"

# Display order (DiffusionFWI sits between TV and RED-DiffEq, DLO last).
METHODS: List[str] = [
    "physical_fwi",
    "tikhonov",
    "tv",
    "diffusion_fwi",
    "red_diffeq",
    "dlo_phase1",
]
DISPLAY_NAMES: Dict[str, str] = {
    "physical_fwi": "Physical FWI",
    "tikhonov": "Tikhonov",
    "tv": "TV",
    "diffusion_fwi": "DiffusionFWI",
    "red_diffeq": "RED-DiffEq",
    "dlo_phase1": "DLO",
}
PLOT_COLORS: Dict[str, str] = {
    "physical_fwi": "#bcbd22",
    "tikhonov": "#888888",
    "tv": "#1f77b4",
    "diffusion_fwi": "#ff7f0e",
    "red_diffeq": "#2ca02c",
    "dlo_phase1": "#8c564b",
}

DATASET_KEYS = {"CurveFault", "CurveVel", "FlatFault", "FlatVel"}


# --- Parse exp.txt ----------------------------------------------------------

def parse_exp_txt(txt_path: Path) -> List[Tuple[str, Path]]:
    """Return list of (dataset_label, scenario_path)."""
    items: List[Tuple[str, Path]] = []
    current_ds = None
    for raw in txt_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in DATASET_KEYS:
            current_ds = line
            continue
        # absolute or repo-relative path → make absolute
        p = Path(line)
        if not p.is_absolute():
            p = REPO_ROOT / p
        items.append((current_ds, p))
    return items


# --- Scenario / title labeling ---------------------------------------------

_SUFFIX_PATTERNS = [
    (re.compile(r"_gauss(\d+\.\d+)$"), lambda m: ("gauss", float(m.group(1)))),
    (re.compile(r"_laplace(\d+\.\d+)$"), lambda m: ("laplace", float(m.group(1)))),
    (re.compile(r"_miss(\d+)$"), lambda m: ("miss", int(m.group(1)))),
]


def scenario_tag(scn_dir: Path) -> str:
    name = scn_dir.name
    for pat, fn in _SUFFIX_PATTERNS:
        m = pat.search(name)
        if m:
            kind, val = fn(m)
            if kind == "gauss":
                return f"Gaussian noise σ={val:.2f}"
            if kind == "laplace":
                return f"Laplacian noise b={val:.2f}"
            if kind == "miss":
                return f"missing {val} traces"
    return "clean"


def dataset_full_name(label: str) -> str:
    return f"{label}-B"


# --- Summary loading --------------------------------------------------------

def load_diffusion_fwi_override(scn_dir: Path) -> Dict | None:
    """Look for a single-method rerun nested directly under scn_dir."""
    for child in sorted(scn_dir.iterdir()):
        if not child.is_dir():
            continue
        s_path = child / "summary.json"
        if not s_path.exists():
            continue
        try:
            d = json.loads(s_path.read_text())
        except json.JSONDecodeError:
            continue
        if "diffusion_fwi" in d:
            return d["diffusion_fwi"]
    return None


def assemble_summary(scn_dir: Path) -> Dict[str, Dict]:
    main_summary = json.loads((scn_dir / "summary.json").read_text())
    diff_fwi_new = load_diffusion_fwi_override(scn_dir)
    if diff_fwi_new is None:
        raise RuntimeError(f"No diffusion_fwi rerun found under {scn_dir}")

    out: Dict[str, Dict] = {}
    for m in METHODS:
        if m == "diffusion_fwi":
            out[m] = diff_fwi_new
        else:
            if m not in main_summary:
                raise RuntimeError(f"Method {m!r} missing in {scn_dir}/summary.json")
            out[m] = main_summary[m]
    return out


# --- Plotting ---------------------------------------------------------------

# Fixed zoom-in baselines for the clean scenarios (std is often large
# enough that adaptive lo-span baselines still hug 0). Tuned so the
# smallest bar is clearly visible while staying below every method's mean.
CLEAN_FIXED_Y0 = {"rmse": 0.10, "mae": 0.05, "ssim": 0.50}


def plot_bars(summary: Dict[str, Dict], dataset_label: str,
              scenario_str: str, out_path: Path, n_samples: int,
              is_clean: bool = False) -> None:
    metrics = [("rmse", "RMSE (norm.)", "lower"),
               ("mae",  "MAE (norm.)",  "lower"),
               ("ssim", "SSIM",         "higher")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    names = [DISPLAY_NAMES[m] for m in METHODS]
    colors = [PLOT_COLORS[m] for m in METHODS]
    x = np.arange(len(METHODS))

    for ax, (key, ylabel, direction) in zip(axes, metrics):
        means = np.array([summary[m][key]["mean"] for m in METHODS])
        stds  = np.array([summary[m][key]["std"]  for m in METHODS])
        lo = float(np.min(means - stds))
        hi = float(np.max(means + stds))
        span_full = max(hi - lo, 1e-6)
        if is_clean:
            # Force a fixed zoom-in baseline (error bars get clipped at y0,
            # which is acceptable — we want bar-height differences visible).
            y0 = CLEAN_FIXED_Y0[key]
        else:
            y0 = lo - 0.20 * span_full
            if key == "ssim":
                y0 = max(y0, -1.0)
            elif y0 < 0:
                y0 = max(0.5 * lo, 1e-3)
        y_top = hi + 0.18 * span_full
        ax.bar(x, means - y0, bottom=y0, yerr=stds, capsize=4, color=colors,
               edgecolor="black", linewidth=0.5, alpha=0.85)
        # value labels above the error bars
        for xi, m_, s_ in zip(x, means, stds):
            ax.text(xi, m_ + s_ + (y_top - y0) * 0.02,
                    f"{m_:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
        ax.set_ylim(y0, y_top)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ({direction} better)")
        ax.grid(axis="y", alpha=0.3)

    cond_str = f"  [{scenario_str}]"
    fig.suptitle(f"{dataset_full_name(dataset_label)}: averaged metrics over "
                 f"{n_samples} samples{cond_str}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    entries = parse_exp_txt(EXP_TXT)
    if not entries:
        raise SystemExit(f"No scenarios parsed from {EXP_TXT}")

    print(f"Processing {len(entries)} scenarios from {EXP_TXT}")
    for dataset, scn_dir in entries:
        if not scn_dir.exists():
            print(f"  SKIP (missing): {scn_dir}")
            continue
        summary = assemble_summary(scn_dir)
        n = summary[METHODS[0]]["rmse"]["n"]
        scn_str = scenario_tag(scn_dir)
        out_path = scn_dir / "bars_v2.png"
        plot_bars(summary, dataset, scn_str, out_path, n_samples=n,
                  is_clean=(scn_str == "clean"))
        print(f"  {dataset:<11s} {scn_str:<28s}  →  {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
