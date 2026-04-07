#!/usr/bin/env python3
"""
Smoke test for CuPy forward + PyTorch wrapper (``src/seismic/wave_equation_forward.py``).

Defaults: logical GPU **cuda:3**; data ``data/Curvevel-B/model60.npy`` and ``data60.npy``.

Example::

  CUDA_DEVICE=0 SAMPLE_INDEX=3 uv run python scripts/test_forward_modeling.py
"""

from __future__ import annotations

import os
import sys

import cupy as cp
import numpy as np
import torch

CUDA_DEVICE = int(os.environ.get("CUDA_DEVICE", "3"))
SAMPLE_INDEX = int(os.environ.get("SAMPLE_INDEX", "0"))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _resolve_data_paths():
    candidates = [
        (
            os.path.join(_ROOT, "data", "Curvevel-B", "model60.npy"),
            os.path.join(_ROOT, "data", "Curvevel-B", "data60.npy"),
        ),
    ]
    for mp, dp in candidates:
        if os.path.isfile(mp) and os.path.isfile(dp):
            return mp, dp
    return candidates[-1]


MODEL_PATH, DATA_PATH = _resolve_data_paths()

from src.seismic import seismic_master_forward_modeling, vel_to_seis


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (CuPy forward runs on GPU).")
    if torch.cuda.device_count() <= CUDA_DEVICE:
        raise RuntimeError(
            f"Need at least {CUDA_DEVICE + 1} GPU(s), visible: {torch.cuda.device_count()}."
            " With CUDA_VISIBLE_DEVICES, logical cuda:3 may not exist."
        )
    torch.cuda.set_device(CUDA_DEVICE)
    cp.cuda.Device(CUDA_DEVICE).use()
    device = torch.device(f"cuda:{CUDA_DEVICE}")
    print(f"device: {device} ({torch.cuda.get_device_name(CUDA_DEVICE)})")

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Place data files at:\n  {MODEL_PATH}\n  {DATA_PATH}"
        )
    velocity_model = np.load(MODEL_PATH)  # (500, 1, 70, 70)
    seismic_obs = np.load(DATA_PATH)  # (500, 5, 1000, 70)
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= velocity_model.shape[0]:
        raise IndexError(f"SAMPLE_INDEX={SAMPLE_INDEX} out of range [0, {velocity_model.shape[0] - 1}]")
    print(f"data: {DATA_PATH}\nmodel: {MODEL_PATH}\nSAMPLE_INDEX: {SAMPLE_INDEX}")

    v = torch.from_numpy(velocity_model[SAMPLE_INDEX, 0].copy()).to(
        device=device, dtype=torch.float64
    )
    print(f"velocity shape={tuple(v.shape)}, [{v.min():.2f}, {v.max():.2f}]")

    out = seismic_master_forward_modeling(v)
    assert out.shape == (5, 1000, 70), out.shape
    assert torch.isfinite(out).all()
    print(f"[OK] seismic_master_forward_modeling: shape={tuple(out.shape)}")
    obs = torch.from_numpy(seismic_obs[SAMPLE_INDEX].astype(np.float64)).to(device)
    mse_obs = torch.nn.functional.mse_loss(out, obs).item()
    print(f"MSE vs observed (data60), sample {SAMPLE_INDEX}: {mse_obs:.6e}")

    v64 = v.detach().double()
    if v64.is_cuda:
        velocity_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(v64))
    else:
        velocity_cp = cp.asarray(v64.cpu().numpy(), dtype=cp.float64)
    min_vel = cp.min(velocity_cp)
    v_vector = cp.concatenate([velocity_cp.flatten(), cp.array([min_vel])]).reshape(-1, 1)
    result, _, _ = vel_to_seis(v_vector)
    seis_cp = result.reshape(5, 1000, 70)
    assert float(cp.abs(seis_cp).max()) < 1e30
    print(f"[OK] vel_to_seis (CuPy): max|u|={float(cp.abs(seis_cp).max()):.4e}")

    vg = v.clone().detach().requires_grad_(True)
    out2 = seismic_master_forward_modeling(vg)
    loss = out2.pow(2).mean()
    loss.backward()
    assert vg.grad is not None and torch.isfinite(vg.grad).all()
    print(f"[OK] backward (CuPy adjoint): loss={loss.item():.6e}, |grad|_mean={vg.grad.abs().mean():.4e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
