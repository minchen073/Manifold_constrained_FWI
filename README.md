# Decoupled Latent Optimization of Diffusion Models for Full Waveform Inversion

Full Waveform Inversion (FWI) with diffusion-model priors: a learned velocity-field
prior (a manifold constraint) is combined with a wave-equation forward operator to
reconstruct subsurface velocity structure from seismic observations.

This repository contains the **main experiments only**: the experiment scripts, the
inversion-method library, and the diffusion-model training code. Exploratory / test
experiments (`exp/`, `experiment/`), datasets (`data/`), and pretrained model weights
(`model.pt`, ~275 MB each, over GitHub's 100 MB per-file limit) are **not tracked** and
must be placed locally.

## Repository layout

```
├── scripts/                     # Main experiment entry points
│   ├── run_4datasets_compare.py   # Single-sample method comparison across 4 datasets
│   ├── run_dataset_average.py     # Multi-sample statistics for one dataset (noise / missing-trace options)
│   ├── replot_4datasets.py        # Re-plot run_4datasets_compare results
│   └── rebuild_dataset_avg_bars.py# Rebuild run_dataset_average bar charts
├── src/
│   ├── methods/                 # Inversion-method library (core)
│   │   └── inversion_methods.py   # Tikhonov / TV / RED-DiffEq / Diffusion-FWI / ILVR / DLO-FWI
│   ├── seismic/                 # Wave-equation forward operator
│   ├── core/                    # SSIM and shared components
│   └── ...                      # cell / dnnlib / torch_utils / utils
├── training/                    # Diffusion-model training code
│   ├── train_ddpm_*.py            # DDPM training for the 4 OpenFWI datasets
│   ├── train_diffusion_v2.py      # v2 training recipe
│   ├── openfwi_unet_wrapper.py    # Model loader/wrapper (used by the experiments)
│   └── configs/                   # Training configs
├── pretrained_model/            # Pretrained DDIM diffusion priors
│   ├── CurveFault-B_DDIM/scheduler_config.json   # only the scheduler config is tracked
│   ├── CurveVel-B_DDIM/                          # model.pt must be placed manually (see below)
│   ├── FlatFault-B_DDIM/
│   └── FlatVel-B_DDIM/
├── pyproject.toml / uv.lock     # Environment
└── .gitignore
```

## Datasets

The experiments use four OpenFWI datasets: **CurveFault-B / CurveVel-B / FlatFault-B /
FlatVel-B**, on a 70×70 velocity grid with 5 sources and 70 receivers. Velocities are
normalized as `(v - 3000) / 1500`, roughly in the range [-1, 1].

> **Note:** the datasets are large (~7 GB) and are **not tracked**. Place them under
> `data/<dataset>/` before running. File names expected by the scripts (see the
> `DATASETS` config inside each script):
> - FlatVel-B / CurveVel-B: `model60.npy` (velocity), `data60.npy` (seismic)
> - FlatFault-B / CurveFault-B: `vel6_1_35.npy`, `seis6_1_35.npy`

## Pretrained models

Each dataset's diffusion-prior weights live at `pretrained_model/<dataset>_DDIM/model.pt`.
Because each file (~275 MB) exceeds GitHub's 100 MB per-file limit, **the weights are not
tracked**; only `scheduler_config.json` is kept in the repo. Download the weights and place
them following the layout below so the scripts can load them:

```
pretrained_model/
├── CurveFault-B_DDIM/model.pt
├── CurveVel-B_DDIM/model.pt
├── FlatFault-B_DDIM/model.pt
└── FlatVel-B_DDIM/model.pt
```

## Inversion methods (`src.methods`)

| Name | Description |
|------|-------------|
| `tikhonov` | Classical Tikhonov-regularized FWI (baseline) |
| `tv` | Total-variation-regularized FWI (baseline) |
| `red_diffeq` | RED-DiffEq: diffusion model as a regularizer |
| `diffusion_fwi` | Diffusion-prior-guided FWI |
| `diffusion_ilvr` | ILVR-style conditional sampling inversion |
| `dlo_fwi` | Decoupled Latent Optimization (DLO) FWI |

## Environment

```bash
uv sync          # install dependencies from pyproject.toml / uv.lock
```
Main dependencies: PyTorch, diffusers, CuPy (the forward operator runs on GPU); a CUDA
environment is required.

## Running the experiments

```bash
# 1) Single-sample method comparison across 4 datasets (results in exp/4ds_compare/<timestamp>/)
python scripts/run_4datasets_compare.py

# 2) Multi-sample statistics for one dataset
python scripts/run_dataset_average.py --dataset FlatFault-B --n_samples 100
#    options: --methods dlo_fwi,red_diffeq   restrict methods
#             --noise_sigma 0.1              noise-robustness test
#             --missing_number 10            missing-trace test
#    results in exp/dataset_avg/<dataset>/<tag>/
```

> Experiment outputs (images, npy, csv) are written to `exp/`, which is ignored by git.
