"""Parametric 2D acoustic wave equation forward + adjoint solver.

CuPy + hand-written CUDA kernels + CUDA Graph capture + hand-written adjoint.
Adapted from src/seismic/wave_equation_forward.py (which is hardcoded to 70x70).

Velocity shape: (nz, nx). Seismic shape: (ns, nt, ng).

Usage:
    engine = WaveEquationForward(nz=70, nx=190, ns=5, nt=1000,
                                  dx=10.0, dt=1e-3, freq=15.0, nbc=120)
    seis = engine(vp_torch)            # torch (nz, nx) -> torch (ns, nt, ng)
    seis.sum().backward()              # autograd works
    vp_torch.grad                      # (nz, nx)
"""

import numpy as np
import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack


# =============================================================================
# Numerical helpers
# =============================================================================

def _unpad_edge_padded_gradient(v_adj: cp.ndarray, nbc: int) -> cp.ndarray:
    """Fold edge-pad contributions back into the inner grid boundaries."""
    H, W = v_adj.shape
    g = v_adj[nbc:-nbc, nbc:-nbc].copy()
    g[0, :] += v_adj[0:nbc, nbc:-nbc].sum(axis=0)
    g[-1, :] += v_adj[-nbc:, nbc:-nbc].sum(axis=0)
    g[:, 0] += v_adj[nbc:-nbc, 0:nbc].sum(axis=1)
    g[:, -1] += v_adj[nbc:-nbc, -nbc:].sum(axis=1)
    g[0, 0] += v_adj[0:nbc, 0:nbc].sum()
    g[0, -1] += v_adj[0:nbc, -nbc:].sum()
    g[-1, 0] += v_adj[-nbc:, 0:nbc].sum()
    g[-1, -1] += v_adj[-nbc:, -nbc:].sum()
    return g


def _ricker(f: float, dt: float, nt: int) -> np.ndarray:
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1
    k = np.arange(1, nw + 1)
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)
    w = np.zeros(nt)
    w[: len(w0)] = w0
    return w


def _abc_coef_2d(Nz: int, Nx: int, nbc: int, dx: float) -> cp.ndarray:
    nz = Nz - 2 * nbc
    nx = Nx - 2 * nbc
    a = (nbc - 1) * dx
    kappa = 3.0 * np.log(1e7) / (2.0 * a)
    damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
    damp = np.zeros((Nz, Nx))
    for iz in range(Nz):
        damp[iz, :nbc] = damp1d[::-1]
        damp[iz, nx + nbc : nx + 2 * nbc] = damp1d
    for ix in range(nbc, nbc + nx):
        damp[:nbc, ix] = damp1d[::-1]
        damp[nz + nbc : nz + 2 * nbc, ix] = damp1d
    return cp.array(damp, dtype=cp.float64)


# =============================================================================
# CUDA kernels — identical to the original implementation, just module-level
# =============================================================================

_KERNEL_LAPG = r'''
extern "C" __global__
void lapg(const double* __restrict__ input,
          double* __restrict__ output,
          const int nx, const int ny,
          const double c2, const double c3) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;
    int ix_p1 = ix+1; if (ix_p1==nx) ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)   ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx) ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)   ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny) iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)   iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny) iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)   iy_m2+=ny;
    double t1 = input[iy*nx + ix_p1] + input[iy*nx + ix_m1]
              + input[iy_p1*nx + ix] + input[iy_m1*nx + ix];
    double t2 = input[iy*nx + ix_p2] + input[iy*nx + ix_m2]
              + input[iy_p2*nx + ix] + input[iy_m2*nx + ix];
    output[idx] = c2*t1 + c3*t2;
}
'''

_KERNEL_UPDATE_P = r'''
extern "C" __global__
void update_p(const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              const double* __restrict__ pout,
              const double* __restrict__ pout1,
              double* __restrict__ pout2,
              const double* __restrict__ lapg_store,
              const int nx, const int ny, const int it,
              const int* __restrict__ src_idx,
              const double* __restrict__ s_mod) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;
    double out = __ldg(&temp1[idx]) * __ldg(&pout1[idx])
               - __ldg(&temp2[idx]) * __ldg(&pout[idx])
               + __ldg(&alpha[idx]) * lapg_store[idx];
    if (idx == src_idx[0]) out += s_mod[it];
    pout2[idx] = out;
}
'''

_KERNEL_UPDATE_P_ADJOINT = r'''
extern "C" __global__
void update_p_adjoint(
    const double* __restrict__ temp1, const double* __restrict__ temp2,
    const double* __restrict__ alpha,
    const double* __restrict__ p_complete1, const double* __restrict__ p_complete2,
    const double* __restrict__ lapg_store,
    double* __restrict__ s_mod_adjoint,
    double* __restrict__ p_complete_adjoint1,
    double* __restrict__ p_complete_adjoint2,
    double* __restrict__ p_complete_adjoint3,
    double* __restrict__ temp1_adjoint, double* __restrict__ temp2_adjoint,
    double* __restrict__ alpha_adjoint, double* __restrict__ lapg_store_adjoint,
    const int nx, const int ny, const int it,
    const double c2, const double c3,
    const int* __restrict__ src_idx) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;
    if (idx == src_idx[0]) s_mod_adjoint[it] = p_complete_adjoint3[idx];
    p_complete_adjoint2[idx] += temp1[idx] * p_complete_adjoint3[idx];
    temp1_adjoint[idx] += p_complete2[idx] * p_complete_adjoint3[idx];
    p_complete_adjoint1[idx] -= temp2[idx] * p_complete_adjoint3[idx];
    temp2_adjoint[idx] -= p_complete1[idx] * p_complete_adjoint3[idx];
    alpha_adjoint[idx] += lapg_store[idx] * p_complete_adjoint3[idx];
    int ix_p1 = ix+1; if (ix_p1==nx) ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)   ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx) ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)   ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny) iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)   iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny) iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)   iy_m2+=ny;
    double t1 = alpha[iy*nx + ix_p1] * p_complete_adjoint3[iy*nx + ix_p1]
              + alpha[iy*nx + ix_m1] * p_complete_adjoint3[iy*nx + ix_m1]
              + alpha[iy_p1*nx + ix] * p_complete_adjoint3[iy_p1*nx + ix]
              + alpha[iy_m1*nx + ix] * p_complete_adjoint3[iy_m1*nx + ix];
    double t2 = alpha[iy*nx + ix_p2] * p_complete_adjoint3[iy*nx + ix_p2]
              + alpha[iy*nx + ix_m2] * p_complete_adjoint3[iy*nx + ix_m2]
              + alpha[iy_p2*nx + ix] * p_complete_adjoint3[iy_p2*nx + ix]
              + alpha[iy_m2*nx + ix] * p_complete_adjoint3[iy_m2*nx + ix];
    p_complete_adjoint2[idx] += c2*t1 + c3*t2;
}
'''

# Compile once at module import.
_lapg = cp.RawModule(code=_KERNEL_LAPG).get_function('lapg')
_update_p = cp.RawModule(code=_KERNEL_UPDATE_P).get_function('update_p')
_update_p_adjoint = cp.RawModule(code=_KERNEL_UPDATE_P_ADJOINT).get_function('update_p_adjoint')


# =============================================================================
# Main solver
# =============================================================================

class WaveEquationForward:
    """Parametric 2D acoustic forward + adjoint solver.

    Memory-checkpointed: forward saves only the seismograms (~ns*nt*ng*8 B);
    backward re-runs the forward wavefield per shot to recover p_complete, so
    peak memory stays around one (nt+2, Nz, Nx) buffer regardless of ns.
    """

    def __init__(self, nz: int, nx: int, ns: int = 5, nt: int = 1000,
                 dx: float = 10.0, dt: float = 1e-3, freq: float = 15.0,
                 nbc: int = 120, source_xs_grid=None, source_z_grid: int = 1,
                 recv_xs_grid=None, recv_z_grid: int = 1):
        self.nz, self.nx = int(nz), int(nx)
        self.ns, self.nt = int(ns), int(nt)
        self.dx, self.dt, self.freq, self.nbc = float(dx), float(dt), float(freq), int(nbc)
        self.Nz, self.Nx = self.nz + 2 * self.nbc, self.nx + 2 * self.nbc

        self.c1 = -2.5
        self.c2 = np.float64(4.0 / 3.0)
        self.c3 = np.float64(-1.0 / 12.0)

        if source_xs_grid is None:
            source_xs_grid = np.linspace(0, self.nx - 1, self.ns).round().astype(int).tolist()
        assert len(source_xs_grid) == self.ns, "source_xs_grid length must equal ns"
        self.source_xs_grid = [int(x) for x in source_xs_grid]
        self.source_z_grid = int(source_z_grid)

        if recv_xs_grid is None:
            recv_xs_grid = np.arange(0, self.nx).tolist()
        self.recv_xs_grid = [int(x) for x in recv_xs_grid]
        self.recv_z_grid = int(recv_z_grid)
        self.ng = len(self.recv_xs_grid)

        self._isx_list = [sx + self.nbc for sx in self.source_xs_grid]
        self._isz = self.source_z_grid + self.nbc
        self._igx_np = (np.array(self.recv_xs_grid) + self.nbc).astype(np.int32)
        self._igz = self.recv_z_grid + self.nbc

        self._src_idx_list = [
            np.int32(self._isz * self.Nx + isx) for isx in self._isx_list
        ]

        s_np = _ricker(self.freq, self.dt, self.nt)
        self._s = cp.array(s_np, dtype=cp.float64)
        self._damp = _abc_coef_2d(self.Nz, self.Nx, self.nbc, self.dx)

        self._allocate_buffers()

        self._stream = cp.cuda.Stream(non_blocking=True)
        self._graph_fwd = None
        self._graph_adj = None

    # ---------- buffer setup ----------
    def _allocate_buffers(self):
        Nz, Nx, nt, ns, ng = self.Nz, self.Nx, self.nt, self.ns, self.ng
        self._v = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._alpha = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._temp1 = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._temp2 = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._s_mod = cp.zeros_like(self._s)

        self._seis = cp.zeros((ns, nt, ng), dtype=cp.float64)
        self._p_complete = cp.zeros((nt + 2, Nz, Nx), dtype=cp.float64)
        self._lapg_store = cp.zeros((nt, Nz, Nx), dtype=cp.float64)

        self._alpha_adj = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._temp1_adj = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._temp2_adj = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._v_adj_buf = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._lapg_store_adj = cp.zeros((Nz, Nx), dtype=cp.float64)
        self._s_mod_adj = cp.zeros_like(self._s)
        self._p_complete_adj = cp.zeros((nt + 2, Nz, Nx), dtype=cp.float64)

        self._p_complete_flat = self._p_complete.ravel()
        self._lapg_store_flat = self._lapg_store.ravel()
        self._temp1_flat = self._temp1.ravel()
        self._temp2_flat = self._temp2.ravel()
        self._alpha_flat = self._alpha.ravel()
        self._p_complete_adj_flat = self._p_complete_adj.ravel()
        self._temp1_adj_flat = self._temp1_adj.ravel()
        self._temp2_adj_flat = self._temp2_adj.ravel()
        self._alpha_adj_flat = self._alpha_adj.ravel()
        self._lapg_store_adj_flat = self._lapg_store_adj.ravel()

        self._igx_dev = cp.array(self._igx_np)
        self._src_idx_dev = cp.zeros((1,), dtype=cp.int32)

    # ---------- velocity prep ----------
    def _prep_velocity(self, vp: cp.ndarray):
        nbc = self.nbc
        self._v[...] = cp.pad(vp, ((nbc, nbc), (nbc, nbc)), mode='edge')
        min_vel = cp.min(vp)
        abc = min_vel * self._damp
        self._alpha[...] = (self._v * (self.dt / self.dx)) ** 2
        kappa = abc * self.dt
        self._temp1[...] = 2 + 2 * self.c1 * self._alpha - kappa
        self._temp2[...] = 1 - kappa
        return min_vel

    # ---------- single-shot forward (writes self._p_complete) ----------
    def _run_one_shot_forward(self, i_shot: int):
        Nz, Nx, nt = self.Nz, self.Nx, self.nt
        tx, ty = 32, 32
        bx = (Nx + tx - 1) // tx
        by = (Nz + ty - 1) // ty

        isx, isz = self._isx_list[i_shot], self._isz
        bdt = (self._v[isz, isx] * self.dt) ** 2
        self._s_mod[...] = bdt * self._s
        self._src_idx_dev[...] = cp.array([self._src_idx_list[i_shot]], dtype=cp.int32)
        self._p_complete[...] = 0

        if self._graph_fwd is None:
            self._stream.begin_capture()
            for it in range(nt):
                _lapg(
                    (bx, by), (tx, ty),
                    (self._p_complete_flat[(it + 1) * (Nz * Nx):],
                     self._lapg_store_flat[(Nz * Nx) * it:],
                     Nx, Nz, self.c2, self.c3),
                )
                _update_p(
                    (bx, by), (tx, ty),
                    (self._temp1_flat, self._temp2_flat, self._alpha_flat,
                     self._p_complete_flat[it * (Nz * Nx):],
                     self._p_complete_flat[(it + 1) * (Nz * Nx):],
                     self._p_complete_flat[(it + 2) * (Nz * Nx):],
                     self._lapg_store_flat[(Nz * Nx) * it:],
                     Nx, Nz, it,
                     self._src_idx_dev, self._s_mod),
                )
            self._graph_fwd = self._stream.end_capture()
            self._graph_fwd.upload(self._stream)

        self._graph_fwd.launch(self._stream)

    def _run_one_shot_adjoint(self, i_shot: int, grad_seis_shot: cp.ndarray):
        Nz, Nx, nt = self.Nz, self.Nx, self.nt
        tx, ty = 32, 32
        bx = (Nx + tx - 1) // tx
        by = (Nz + ty - 1) // ty

        self._p_complete_adj[...] = 0
        self._p_complete_adj[2 : nt + 2, self._igz, self._igx_dev] = grad_seis_shot
        self._s_mod_adj[...] = 0
        self._lapg_store_adj[...] = 0

        if self._graph_adj is None:
            self._stream.begin_capture()
            for it in range(nt - 1, -1, -1):
                _update_p_adjoint(
                    (bx, by), (tx, ty),
                    (self._temp1_flat, self._temp2_flat, self._alpha_flat,
                     self._p_complete_flat[it * (Nz * Nx):],
                     self._p_complete_flat[(it + 1) * (Nz * Nx):],
                     self._lapg_store_flat[it * (Nz * Nx):],
                     self._s_mod_adj,
                     self._p_complete_adj_flat[it * (Nz * Nx):],
                     self._p_complete_adj_flat[(it + 1) * (Nz * Nx):],
                     self._p_complete_adj_flat[(it + 2) * (Nz * Nx):],
                     self._temp1_adj_flat, self._temp2_adj_flat,
                     self._alpha_adj_flat, self._lapg_store_adj_flat,
                     Nx, Nz, it, self.c2, self.c3,
                     self._src_idx_dev),
                )
            self._graph_adj = self._stream.end_capture()
            self._graph_adj.upload(self._stream)

        self._graph_adj.launch(self._stream)

    # ---------- public CuPy API ----------
    def forward_cp(self, vp: cp.ndarray) -> cp.ndarray:
        assert vp.shape == (self.nz, self.nx), \
            f"vp shape {tuple(vp.shape)} != {(self.nz, self.nx)}"
        self._prep_velocity(vp)
        cp.cuda.Stream.null.synchronize()
        with self._stream:
            for i_shot in range(self.ns):
                self._run_one_shot_forward(i_shot)
                self._stream.synchronize()
                self._seis[i_shot, ...] = self._p_complete[
                    2 : self.nt + 2, self._igz, self._igx_dev
                ]
        self._stream.synchronize()
        return self._seis.copy()

    def backward_cp(self, vp: cp.ndarray, grad_seis: cp.ndarray) -> cp.ndarray:
        """Re-runs forward per shot (checkpointing) to recover wavefield, then adjoint."""
        assert vp.shape == (self.nz, self.nx)
        assert grad_seis.shape == (self.ns, self.nt, self.ng)

        self._prep_velocity(vp)
        self._alpha_adj[...] = 0
        self._temp1_adj[...] = 0
        self._temp2_adj[...] = 0
        self._v_adj_buf[...] = 0

        cp.cuda.Stream.null.synchronize()
        with self._stream:
            for i_shot in range(self.ns):
                self._run_one_shot_forward(i_shot)
                self._stream.synchronize()
                self._run_one_shot_adjoint(i_shot, grad_seis[i_shot])
                self._stream.synchronize()
                isx, isz = self._isx_list[i_shot], self._isz
                self._v_adj_buf[isz, isx] += (
                    2 * self.dt ** 2 * self._v[isz, isx]
                    * cp.sum(self._s_mod_adj * self._s)
                )
        self._stream.synchronize()

        kappa_adj = -self._temp2_adj
        alpha_adj_full = self._alpha_adj + 2 * self.c1 * self._temp1_adj
        kappa_adj = kappa_adj + (-self._temp1_adj)
        abc_adj = kappa_adj * self.dt
        v2_adj = alpha_adj_full * (self.dt / self.dx) ** 2
        v_adj_padded = self._v_adj_buf + 2 * self._v * v2_adj
        min_vel_adj = cp.sum(abc_adj * self._damp)
        vv_adj = _unpad_edge_padded_gradient(v_adj_padded, self.nbc)
        idx_min = cp.unravel_index(cp.argmin(vp), vp.shape)
        vv_adj[idx_min[0], idx_min[1]] += min_vel_adj
        return vv_adj

    # ---------- public PyTorch API ----------
    def __call__(self, vp_torch: torch.Tensor) -> torch.Tensor:
        return _WaveEqFn.apply(vp_torch, self)


class _WaveEqFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vp_torch: torch.Tensor, engine: WaveEquationForward):
        if vp_torch.shape != (engine.nz, engine.nx):
            raise ValueError(
                f"velocity shape must be {(engine.nz, engine.nx)}, got {tuple(vp_torch.shape)}"
            )

        if vp_torch.device.type == 'cuda':
            cp.cuda.Device(vp_torch.device.index or 0).use()
            vp_cp = cp.from_dlpack(to_dlpack(vp_torch.detach().double().contiguous()))
        else:
            vp_cp = cp.asarray(vp_torch.detach().cpu().numpy(), dtype=cp.float64)

        seis_cp = engine.forward_cp(vp_cp)
        ctx.engine = engine
        ctx.save_for_backward(vp_torch)

        if vp_torch.device.type == 'cuda':
            seis_torch = from_dlpack(seis_cp.toDlpack())
        else:
            seis_torch = torch.from_numpy(cp.asnumpy(seis_cp)).to(vp_torch.device)
        return seis_torch.to(vp_torch.dtype)

    @staticmethod
    def backward(ctx, grad_seis: torch.Tensor):
        engine = ctx.engine
        (vp_torch,) = ctx.saved_tensors
        target = (engine.ns, engine.nt, engine.ng)
        if grad_seis.shape != target:
            raise ValueError(
                f"grad seismic shape must be {target}, got {tuple(grad_seis.shape)}"
            )

        if grad_seis.device.type == 'cuda':
            cp.cuda.Device(grad_seis.device.index or 0).use()
            grad_cp = cp.from_dlpack(to_dlpack(grad_seis.detach().double().contiguous()))
            vp_cp = cp.from_dlpack(to_dlpack(vp_torch.detach().double().contiguous()))
        else:
            grad_cp = cp.asarray(grad_seis.detach().cpu().numpy(), dtype=cp.float64)
            vp_cp = cp.asarray(vp_torch.detach().cpu().numpy(), dtype=cp.float64)

        vp_grad_cp = engine.backward_cp(vp_cp, grad_cp)

        if grad_seis.device.type == 'cuda':
            vp_grad_torch = from_dlpack(vp_grad_cp.toDlpack())
        else:
            vp_grad_torch = torch.from_numpy(cp.asnumpy(vp_grad_cp)).to(grad_seis.device)
        return vp_grad_torch.to(vp_torch.dtype), None
