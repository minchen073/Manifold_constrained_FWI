#!/usr/bin/env python3
"""
2D acoustic forward modeling (CuPy/CUDA) with a PyTorch tensor wrapper (DLPack).

Merged from ``forward_modeling.py`` and ``seismic_master_forward_wrapper.py`` for a single
module under ``src/seismic``.

Public PyTorch entry points ``seismic_master_forward_modeling`` / ``torch_forward_modeling_gpu``:
input velocity ``(70, 70)``, output seismograms ``(5, 1000, 70)``; wavelet, grid, sources and
receivers are fixed by module-level globals (no dt/dx/freq arguments in the API).
"""

import numpy as np
import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

# seis_numerics functions integrated below

# =============================================================================
# Numerical Helper Functions (from seis_numerics.py)
# =============================================================================

def unpad_edge_padded_gradient(v_adjoint: cp.ndarray, nbc: int) -> cp.ndarray:
    """
    Given a gradient array `v_adjoint` that was computed on an edge-padded field (mode='edge'),
    fold (sum) all contributions from the padded border back into the corresponding boundary cells
    of the unpadded region, then return the unpadded result.

    Parameters
    ----------
    v_adjoint : cp.ndarray
        A 2D CuPy array of shape (H, W), where H = N + 2*nbc and W = M + 2*nbc. It represents
        the gradient w.r.t. a field that was originally of shape (N, M) and then padded by `nbc`
        on each side with mode='edge'.
    nbc : int
        Number of boundary‐padding cells on each side.

    Returns
    -------
    cp.ndarray
        A 2D CuPy array of shape (N, M), which is the gradient folded back into the original
        (unpadded) domain.
    """
    # Full padded gradient array
    W_full = v_adjoint
    H, W = W_full.shape

    # Dimensions of the unpadded region
    N = H - 2 * nbc
    M = W - 2 * nbc

    # Extract the interior (unpadded) slice
    g = W_full[nbc:-nbc, nbc:-nbc].copy()  # shape = (N, M)

    # Top edge padding → fold into row 0 of `g`
    top_block = W_full[0:nbc, nbc:-nbc]        # shape = (nbc, M)
    g[0, :] += top_block.sum(axis=0)

    # Bottom edge padding → fold into row N-1 of `g`
    bottom_block = W_full[-nbc:, nbc:-nbc]      # shape = (nbc, M)
    g[-1, :] += bottom_block.sum(axis=0)

    # Left edge padding → fold into column 0 of `g`
    left_block = W_full[nbc:-nbc, 0:nbc]        # shape = (N, nbc)
    g[:, 0] += left_block.sum(axis=1)

    # Right edge padding → fold into column M-1 of `g`
    right_block = W_full[nbc:-nbc, -nbc:]       # shape = (N, nbc)
    g[:, -1] += right_block.sum(axis=1)

    # Four corner blocks → fold into the four corners of `g`
    # Top-left corner
    corner_tl = W_full[0:nbc, 0:nbc]            # shape = (nbc, nbc)
    g[0, 0] += corner_tl.sum()

    # Top-right corner
    corner_tr = W_full[0:nbc, -nbc:]            # shape = (nbc, nbc)
    g[0, -1] += corner_tr.sum()

    # Bottom-left corner
    corner_bl = W_full[-nbc:, 0:nbc]            # shape = (nbc, nbc)
    g[-1, 0] += corner_bl.sum()

    # Bottom-right corner
    corner_br = W_full[-nbc:, -nbc:]            # shape = (nbc, nbc)
    g[-1, -1] += corner_br.sum()

    return g

# =============================================================================
# Seismic Forward Modeling Implementation (from seis_forward2.py)
# =============================================================================

# Set up stuff for CUDA graphing
stream = cp.cuda.Stream(non_blocking=True)
graph = 0
graph_diff = 0
graph_adjoint = 0

# Precalculations, including preallocating matrices
def ricker(f, dt, nt=None):
    """
    Ricker wavelet in time.

    There is no separate ``peak_time`` parameter: length and shape follow ``f`` and ``dt``;
    ``expand_source`` places the wavelet at the first ``len(w0)`` samples starting at t=0.
    The main lobe peak lies inside ``w0`` (roughly mid-segment); it is not an independent delay.
    """
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1
    k = np.arange(1, nw + 1)  
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)    
    if nt is not None:
        if nt < len(w0):
            raise ValueError("nt is smaller than condition!")
        w = np.zeros(nt)  
        w[0:len(w0)] = w0
    else:
        w = np.zeros(len(w0))
        w[0:] = w0
    if nt is not None:
        tw = np.arange(1, len(w)) * dt
    else:
        tw = np.arange(1, len(w)) * dt
    return w, tw

def expand_source(s0, nt):
    s0 = np.asarray(s0).flatten()
    s = np.zeros(nt)
    s[0:len(s0)] = s0
    return s

def AbcCoef2D(nzbc, nxbc, nbc, dx):
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc
    a = (nbc - 1) * dx
    kappa = 3.0 * np.log(1e7) / (2.0 * a)
    damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
    damp = np.zeros((nzbc, nxbc))
    for iz in range(nzbc):
        damp[iz, :nbc] = damp1d[::-1]
        damp[iz, nx + nbc : nx + 2 * nbc] = damp1d
    for ix in range(nbc, nbc + nx):
        damp[:nbc, ix] = damp1d[::-1]
        damp[nz + nbc: nz + 2 * nbc, ix] = damp1d
    return cp.array(damp, dtype=cp.float64)

def adjust_sr(coord, dx, nbc):
    isx = int(round(coord['sx'] / dx)) + nbc
    isz = int(round(coord['sz'] / dx)) + nbc
    igx = (np.round(np.array(coord['gx']) / dx) + nbc).astype(int)
    igz = (np.round(np.array(coord['gz']) / dx) + nbc).astype(int)
    if abs(coord['sz']) < 0.5:
        isz += 1
    igz = igz + (np.abs(np.array(coord['gz'])) < 0.5).astype(int)
    return isx, isz, igx, igz

# Global parameters
nz = 70
nx = 70
dx = 10
nbc = 120
nt = 1000
dt = (1e-3)
freq = 15
s, _ = ricker(freq, dt)
s = expand_source(s, nt)
s = cp.array(s, dtype=cp.float64)
c1 = (-2.5)
c2 = (4.0 / 3.0)
c3 = (-1.0 / 12.0)
c2, c3 = np.array(c2, dtype=np.float64), np.array(c3, dtype=np.float64)

# Source and receiver setup
src_idx_list = []
isx_list = []
isz_list = []
for i_source in range(5):
    coord = {}
    source_x = [0, 17, 34, 52, 69][i_source]
    coord['sx'] = source_x * dx        
    coord['sz'] = 1 * dx
    coord['gx'] = np.arange(0, nx) * dx
    coord['gz'] = np.ones_like(coord['gx']) * dx
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)
    src_idx = np.int32(isz*310 + isx)
    src_idx_list.append(src_idx)
    isx_list.append(isx)
    isz_list.append(isz)

ng = len(coord['gx'])
damp = AbcCoef2D(310, 310, nbc, dx)
nx, nz = 310, 310
rcv_idx = nx*igz+igx

# Prepare base matrices
seis_combined = None
p_complete = None
lapg_store = None
temp1 = cp.zeros((nx, nz), dtype=cp.float64)
temp2 = cp.zeros((nx, nz), dtype=cp.float64)
alpha = cp.zeros((nx, nz), dtype=cp.float64)
v = cp.zeros((nx, nz), dtype=cp.float64)
temp1_flat = temp1.ravel()
temp2_flat = temp2.ravel()
alpha_flat = alpha.ravel()        
s_mod = cp.zeros_like(s)

# Prepare forward propagation matrices
seis_combined_diff = None
p_complete_diff = None
lapg_store_diff = cp.zeros((nx, nz), dtype=cp.float64)
temp1_diff = cp.zeros((nx, nz), dtype=cp.float64)
temp2_diff = cp.zeros((nx, nz), dtype=cp.float64)
alpha_diff = cp.zeros((nx, nz), dtype=cp.float64)
v_diff = cp.zeros((nx, nz), dtype=cp.float64)
s_mod_diff = cp.zeros_like(s)
lapg_store_diff_flat = lapg_store_diff.ravel()
temp1_diff_flat = temp1_diff.ravel()
temp2_diff_flat = temp2_diff.ravel()
alpha_diff_flat = alpha_diff.ravel()
v_diff_flat = v_diff.ravel()

# Prepare backward propagation matrices
seis_combined_adjoint = None
p_complete_adjoint = None
lapg_store_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
temp1_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
temp2_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
alpha_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
v_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
s_mod_adjoint = cp.zeros_like(s)
lapg_store_adjoint_flat = lapg_store_adjoint.ravel()
temp1_adjoint_flat = temp1_adjoint.ravel()
temp2_adjoint_flat = temp2_adjoint.ravel()
alpha_adjoint_flat = alpha_adjoint.ravel()
v_adjoint_flat = v_adjoint.ravel()

src_idx_dev = cp.zeros((1,), dtype=cp.int32)
igz_dev, igx_dev = cp.array(igz), cp.array(igx)

# CUDA kernel code
kernel_code = r'''
extern "C" __global__
void lapg(
              const double* __restrict__ input,
              double* __restrict__ output,
              const int    nx,
              const int    ny,
              const double  c2,
              const double  c3) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    // Manual wrap at +-1, +-2
    int ix_p1 = ix+1; if (ix_p1==nx)  ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)    ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx)  ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)     ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny)  iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)    iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny)  iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)     iy_m2+=ny;

    double t1;
    double t2;
    
    // Collect neighbors (+-1)
    t1 = input[iy  * nx + ix_p1]
             + input[iy  * nx + ix_m1]
             + input[iy_p1 * nx + ix  ]
             + input[iy_m1 * nx + ix  ];
    // Collect neighbors (+-2)
    t2 = input[iy  * nx + ix_p2]
             + input[iy  * nx + ix_m2]
             + input[iy_p2 * nx + ix  ]
             + input[iy_m2 * nx + ix  ];

    
    output[idx] = c2*t1+c3*t2;
}
'''

module = cp.RawModule(code=kernel_code)
lapg = module.get_function('lapg')

# CUDA kernel to update p and add source
kernel_code = r'''
extern "C" __global__
void update_p(
              const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              const double*             __restrict__ pout,
              const double*             __restrict__ pout1,
              double*             __restrict__ pout2,
              const double*             __restrict__ lapg_store,
              const int    nx,
              const int    ny,
              const int    it,
              const int*   __restrict__ src_idx,
              const double* __restrict__ s_mod) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    double temp1_local = __ldg(&temp1[idx]);
    double temp2_local = __ldg(&temp2[idx]);
    double pout1_local = __ldg(&pout1[idx]);
    double pout_local = __ldg(&pout[idx]);
    double alpha_local = __ldg(&alpha[idx]);
    double out = temp1_local*pout1_local
              - temp2_local*pout_local
              + alpha_local*lapg_store[idx];

    // fused source injection:
    if (idx == src_idx[0]) {
        out += s_mod[it];
    }
    pout2[idx] = out;
}
'''

module = cp.RawModule(code=kernel_code)
update_p = module.get_function('update_p')

# Forward propagation of update_p
kernel_code = r'''
extern "C" __global__
void update_p_diff(
              const double* __restrict__ temp1,
              const double* __restrict__ temp1_diff,
              const double* __restrict__ temp2,
              const double* __restrict__ temp2_diff,
              const double* __restrict__ alpha,
              const double* __restrict__ alpha_diff,
              const double*             __restrict__ pout,
              const double*             __restrict__ pout1,
              const double*             __restrict__ pout2,
              const double*             __restrict__ lapg_store,
              const double*             __restrict__ lapg_store_diff,
              double*  __restrict__ pout_diff,
              double*  __restrict__ pout1_diff,
              double*  __restrict__ pout2_diff,
              const int    nx,
              const int    ny,
              const int it,
              const int*   __restrict__ src_idx,
              const double* __restrict__ s_mod_diff) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    double out = temp1_diff[idx] * pout1[idx] + temp1[idx]*pout1_diff[idx]-temp2_diff[idx] * pout[idx] - temp2[idx]*pout_diff[idx] +
        alpha_diff[idx]*lapg_store[idx] + alpha[idx]*lapg_store_diff[idx];

    // fused source injection:
    if (idx == src_idx[0]) {
        out += s_mod_diff[it];
    }
    pout2_diff[idx] = out;
}
'''

module = cp.RawModule(code=kernel_code)
update_p_diff = module.get_function('update_p_diff')

# Backward propagation of update_p (with lapg also folded in)
kernel_code = r'''
extern "C" __global__
void update_p_adjoint(
              const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              const double* __restrict__ p_complete1,
              const double* __restrict__ p_complete2,
              const double* __restrict__ lapg_store,
              double*  __restrict__ s_mod_adjoint,
              double*  __restrict__ p_complete_adjoint1,
              double*  __restrict__ p_complete_adjoint2,
              double*  __restrict__ p_complete_adjoint3,
              double*  __restrict__ temp1_adjoint,
              double*  __restrict__ temp2_adjoint,
              double*  __restrict__ alpha_adjoint,
              double*  __restrict__ lapg_store_adjoint,
              const int    nx,
              const int    ny,
              const int    it,
              const double  c2,
              const double  c3,
              const int*   __restrict__ src_idx) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    if (idx == src_idx[0]) {
        s_mod_adjoint[it] = p_complete_adjoint3[idx];
    }

    p_complete_adjoint2[idx] += temp1[idx]*p_complete_adjoint3[idx];
    temp1_adjoint[idx] += p_complete2[idx]*p_complete_adjoint3[idx];
    p_complete_adjoint1[idx] -= temp2[idx]*p_complete_adjoint3[idx];
    temp2_adjoint[idx] -= p_complete1[idx]*p_complete_adjoint3[idx];
    alpha_adjoint[idx] += lapg_store[idx] * p_complete_adjoint3[idx];

    // Manual wrap at +-1, +-2
    int ix_p1 = ix+1; if (ix_p1==nx)  ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)    ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx)  ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)     ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny)  iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)    iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny)  iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)     iy_m2+=ny;

    // Collect neighbors (+-1)
    double t1 = alpha[iy  * nx + ix_p1] * p_complete_adjoint3[0 + iy  * nx + ix_p1] +
                alpha[iy  * nx + ix_m1] * p_complete_adjoint3[0 + iy  * nx + ix_m1] +
                alpha[iy_p1  * nx + ix] * p_complete_adjoint3[0 + iy_p1  * nx + ix] +
                alpha[iy_m1  * nx + ix] * p_complete_adjoint3[0 + iy_m1  * nx + ix];
    // Collect neighbors (+-2)
    double t2 = alpha[iy  * nx + ix_p2] * p_complete_adjoint3[0 + iy  * nx + ix_p2] +
                alpha[iy  * nx + ix_m2] * p_complete_adjoint3[0 + iy  * nx + ix_m2] +
                alpha[iy_p2  * nx + ix] * p_complete_adjoint3[0 + iy_p2  * nx + ix] +
                alpha[iy_m2  * nx + ix] * p_complete_adjoint3[0 + iy_m2  * nx + ix];

    p_complete_adjoint2[idx]+=c2*t1+c3*t2;
}
'''

module = cp.RawModule(code=kernel_code)
update_p_adjoint = module.get_function('update_p_adjoint')

def prep_run(vec):
    vv = cp.reshape(vec[:-1, 0], (70, 70))
    min_vel = vec[-1, 0]
    
    v[...] = cp.pad(vv, ((nbc, nbc), (nbc, nbc)), mode='edge')
    abc = min_vel*damp

    alpha[...] = (v * (dt / dx)) ** 2    
    kappa = abc * dt
    temp1[...] = 2 + 2 * c1 * alpha - kappa
    temp2[...] = 1 - kappa

def prep_run_diff(vec_diff):
    vv = cp.reshape(vec_diff[:-1, 0], (70, 70))
    min_vel_diff = vec_diff[-1, 0]
    
    v_diff[...] = cp.pad(vv, ((nbc, nbc), (nbc, nbc)), mode='edge')
    abc_diff = min_vel_diff*damp

    alpha_diff[...] = v_diff * v * (2*(dt / dx) **2)
    kappa_diff = abc_diff * dt
    temp1_diff[...] = 2 * c1 * alpha_diff - kappa_diff
    temp2_diff[...] = - kappa_diff

    return v_diff, temp1_diff, temp2_diff, alpha_diff

def prep_run_adjoint():
    kappa_adjoint = -temp2_adjoint
    alpha_adjoint[...] += 2 * c1 * temp1_adjoint
    kappa_adjoint += -temp1_adjoint
    abc_adjoint = kappa_adjoint * dt
    v2_adjoint = alpha_adjoint * (dt/dx)**2

    v_adjoint[...] += 2*v*v2_adjoint
    min_vel_adjoint = cp.sum(abc_adjoint*damp)
    vv_adjoint = unpad_edge_padded_gradient(v_adjoint, nbc)

    result_adjoint = cp.zeros((4901, 1), dtype=cp.float64)
    result_adjoint[-1, 0] = min_vel_adjoint
    result_adjoint[:-1, 0] = vv_adjoint.flatten()

    return result_adjoint

def vel_to_seis(vec, vec_diff=None, vec_adjoint=None, adjoint_on_residual=False):
    """
    Main forward modeling function
    """
    # Input checks
    assert vec.shape == (4901, 1)
    assert vec_adjoint is None or vec_adjoint.shape == (5*1000*70, 1)
    assert vec_diff is None or vec_diff.shape == (4901, 1)
    do_diff = not (vec_diff is None)
    do_adjoint = not (vec_adjoint is None)
    assert vec.dtype == cp.float64
    assert vec_diff is None or vec_diff.dtype == cp.float64
    assert vec_adjoint is None or vec_adjoint.dtype == cp.float64

    # Preallocate matrices not done above
    global seis_combined, p_complete, lapg_store, seis_combined_diff, p_complete_diff, seis_combined_adjoint, p_complete_adjoint
    global p_complete_flat, p_complete_diff_flat, p_complete_adjoint_flat, lapg_store_flat
    global temp1, temp2, alpha, v, s_mod, temp1_diff, temp2_diff, alpha_diff, v_diff, s_mod_diff
    global temp1_adjoint, temp2_adjoint, alpha_adjoint, v_adjoint, s_mod_adjoint
    global lapg_store_diff, lapg_store_adjoint
    global s, damp, src_idx_dev, igz_dev, igx_dev
    global temp1_flat, temp2_flat, alpha_flat
    global lapg_store_diff_flat, temp1_diff_flat, temp2_diff_flat, alpha_diff_flat, v_diff_flat
    global lapg_store_adjoint_flat, temp1_adjoint_flat, temp2_adjoint_flat, alpha_adjoint_flat, v_adjoint_flat
    global stream, graph, graph_adjoint, graph_diff
    
    # Re-init globals if the active CuPy device changed
    need_reinit = False
    current_device_id = cp.cuda.Device().id
    
    if seis_combined is not None:
        try:
            global_device_id = seis_combined.device.id
            if global_device_id != current_device_id:
                need_reinit = True
        except (AttributeError, RuntimeError, cp.cuda.runtime.CUDARuntimeError):
            need_reinit = True

    if not need_reinit:
        try:
            test_array = cp.zeros(1, dtype=cp.float64)
            with stream:
                _ = test_array + 1
            del test_array
        except (RuntimeError, cp.cuda.runtime.CUDARuntimeError):
            need_reinit = True

    if seis_combined is None or need_reinit:
        s_np, _ = ricker(freq, dt)
        s_np = expand_source(s_np, nt)
        s = cp.array(s_np, dtype=cp.float64)
        damp = AbcCoef2D(310, 310, nbc, dx)

        seis_combined = cp.zeros((5, 1000, 70), dtype=cp.float64)
        p_complete = cp.zeros((nt+2, nx, nz), dtype=cp.float64)
        lapg_store = cp.zeros((nt, nx, nz), dtype=cp.float64)
        seis_combined_diff = cp.zeros((5, 1000, 70), dtype=cp.float64)
        p_complete_diff = cp.zeros((nt+2, nx, nz), dtype=cp.float64)
        seis_combined_adjoint = cp.zeros((5, 1000, 70), dtype=cp.float64)
        p_complete_adjoint = cp.zeros((nt+2, nx, nz), dtype=cp.float64)
        
        p_complete_flat = p_complete.ravel()
        lapg_store_flat = lapg_store.ravel()
        p_complete_diff_flat = p_complete_diff.ravel()
        p_complete_adjoint_flat = p_complete_adjoint.ravel()

        temp1 = cp.zeros((nx, nz), dtype=cp.float64)
        temp2 = cp.zeros((nx, nz), dtype=cp.float64)
        alpha = cp.zeros((nx, nz), dtype=cp.float64)
        v = cp.zeros((nx, nz), dtype=cp.float64)
        s_mod = cp.zeros_like(s)
        lapg_store_diff = cp.zeros((nx, nz), dtype=cp.float64)
        temp1_diff = cp.zeros((nx, nz), dtype=cp.float64)
        temp2_diff = cp.zeros((nx, nz), dtype=cp.float64)
        alpha_diff = cp.zeros((nx, nz), dtype=cp.float64)
        v_diff = cp.zeros((nx, nz), dtype=cp.float64)
        s_mod_diff = cp.zeros_like(s)
        lapg_store_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
        temp1_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
        temp2_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
        alpha_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
        v_adjoint = cp.zeros((nx, nz), dtype=cp.float64)
        s_mod_adjoint = cp.zeros_like(s)

        temp1_flat = temp1.ravel()
        temp2_flat = temp2.ravel()
        alpha_flat = alpha.ravel()
        lapg_store_diff_flat = lapg_store_diff.ravel()
        temp1_diff_flat = temp1_diff.ravel()
        temp2_diff_flat = temp2_diff.ravel()
        alpha_diff_flat = alpha_diff.ravel()
        v_diff_flat = v_diff.ravel()
        lapg_store_adjoint_flat = lapg_store_adjoint.ravel()
        temp1_adjoint_flat = temp1_adjoint.ravel()
        temp2_adjoint_flat = temp2_adjoint.ravel()
        alpha_adjoint_flat = alpha_adjoint.ravel()
        v_adjoint_flat = v_adjoint.ravel()

        src_idx_dev = cp.zeros((1,), dtype=cp.int32)
        igz_dev, igx_dev = cp.array(igz), cp.array(igx)

        stream = cp.cuda.Stream(non_blocking=True)
        graph = 0
        graph_diff = 0
        graph_adjoint = 0
    
    # PREPARATION
    prep_run(vec)
     
    if do_diff:
        prep_run_diff(vec_diff)        

    if do_adjoint:        
        alpha_adjoint[...] = 0
        temp1_adjoint[...] = 0
        temp2_adjoint[...] = 0
        v_adjoint[...] = 0
        seis_combined_adjoint[...] = cp.reshape(vec_adjoint, (5, 1000, 70))

    tx, ty = 32, 32
    bx = (nx + tx - 1) // tx
    by = (nz + ty - 1) // ty  

    # LOOP

    cp.cuda.Stream.null.synchronize()

    with stream:        
        for i_source in range(5):     
            ## Calculate seismogram
            # Prepare source injection
            src_idx = src_idx_list[i_source]
            bdt = (v[isz_list[i_source], isx_list[i_source]]*dt)**2
            s_mod[...] = bdt*s            
            src_idx_dev[...] = cp.array(src_idx, dtype=cp.int32)

            # We capture the full time loop as a CUDA graph.
            if graph==0:
                stream.begin_capture()
                for it in range(0, nt):
                    lapg(
                        (bx, by), (tx, ty),
                        (p_complete_flat[(it+1)*(nx*nz):],
                        lapg_store_flat[nx*nz*it:],
                        nx, nz,
                        c2, c3,
                        ))     
                    update_p(
                                (bx, by), (tx, ty),
                                (
                                    temp1_flat, temp2_flat, alpha_flat,
                                    p_complete_flat[it*(nx*nz):],
                                    p_complete_flat[(it+1)*(nx*nz):],
                                    p_complete_flat[(it+2)*(nx*nz):],
                                    lapg_store_flat[nx*nz*it:],
                                    nx, nz, it,
                                    src_idx_dev, s_mod
                                )
                            )
    
                graph = stream.end_capture()
                graph.upload(stream)
            graph.launch(stream)
            stream.synchronize()
            # p_complete[2:1002] -> 1000 time steps
            seis_combined[i_source, ...] = p_complete[2:1002, igz_dev, igx_dev]

            ## Forward propagation
            if do_diff:                
                bdt_diff = 2*((v[isz_list[i_source], isx_list[i_source]]*v_diff[isz_list[i_source], isx_list[i_source]]))* dt**2
                s_mod_diff[...] = bdt_diff*s
                if graph_diff==0:
                    stream.begin_capture()                    
                    for it in range(0, nt):
                        lapg(
                            (bx, by), (tx, ty),
                            (p_complete_diff_flat[(it+1)*(nx*nz):],
                            lapg_store_diff,
                            nx, nz,
                            c2, c3,
                            ))
                        update_p_diff(
                                    (bx, by), (tx, ty),
                                    (
                                        temp1_flat, temp1_diff_flat, temp2_flat, temp2_diff_flat, alpha_flat, alpha_diff_flat, 
                                        p_complete_flat[it*(nx*nz):],
                                        p_complete_flat[(it+1)*(nx*nz):],
                                        p_complete_flat[(it+2)*(nx*nz):],
                                        lapg_store_flat[nx*nz*it:], lapg_store_diff_flat,
                                        p_complete_diff_flat[it*(nx*nz):],
                                        p_complete_diff_flat[(it+1)*(nx*nz):],
                                        p_complete_diff_flat[(it+2)*(nx*nz):],
                                        nx, nz, it,
                                        src_idx_dev, s_mod_diff
                                    )
                                )
                    graph_diff = stream.end_capture()
                    graph_diff.upload(stream)       
                graph_diff.launch(stream)
                seis_combined_diff[i_source, ...] = p_complete_diff[2:1002, igz_dev, igx_dev]
            # Backward propagation
            if do_adjoint:     
                p_complete_adjoint[...] = 0
                if adjoint_on_residual:
                    p_complete_adjoint[2:1002, igz_dev, igx_dev] = seis_combined[i_source, ...]-seis_combined_adjoint[i_source, ...]
                else:
                    p_complete_adjoint[2:1002, igz_dev, igx_dev] = seis_combined_adjoint[i_source, ...]
                s_mod_adjoint[...] = 0
                if graph_adjoint==0:
                    stream.begin_capture()
                    for it in np.arange(nt-1, -1, -1):                               
                        update_p_adjoint(
                                (bx, by), (tx, ty),
                                (
                                    temp1_flat, temp2_flat, alpha_flat,
                                    p_complete_flat[(it)*(nx*nz):], p_complete_flat[(it+1)*(nx*nz):],
                                    lapg_store_flat[(it)*(nx*nz):],
                                    s_mod_adjoint,  
                                    p_complete_adjoint_flat[(it)*(nx*nz):],
                                    p_complete_adjoint_flat[(it+1)*(nx*nz):],
                                    p_complete_adjoint_flat[(it+2)*(nx*nz):],
                                    temp1_adjoint_flat, temp2_adjoint_flat, alpha_adjoint_flat, lapg_store_adjoint_flat,                                    
                                    nx, nz, it,
                                    c2, c3,
                                    src_idx_dev
                                )
                            )
                    graph_adjoint = stream.end_capture()
                    graph_adjoint.upload(stream)  
                graph_adjoint.launch(stream)
                v_adjoint[isz_list[i_source], isx_list[i_source]] += 2*dt**2 * v[isz_list[i_source], isx_list[i_source]] * cp.sum(s_mod_adjoint*s)
    
    stream.synchronize()
            
    # FINALIZE
    assert seis_combined.shape == (5, 1000, 70)
    result = cp.copy(seis_combined.flatten()[:, None])
    if do_diff:
        assert seis_combined_diff.shape == (5, 1000, 70)
        result_diff = cp.copy(seis_combined_diff.flatten()[:, None])
    else:
        result_diff = None
    if do_adjoint:
        result_adjoint = cp.copy(prep_run_adjoint())
    else:
        result_adjoint = None

    return result, result_diff, result_adjoint


# --- PyTorch wrapper (vel_to_seis above) ---
# Geometry and dt/dx/freq/nt/nbc are module globals; API takes a (70, 70) velocity field only.

_VELOCITY_SHAPE = (70, 70)


class SeismicMasterForwardModelingFunction(torch.autograd.Function):
    """Wraps ``vel_to_seis``; forward settings are module-level globals."""

    @staticmethod
    def forward(ctx, velocity_tensor: torch.Tensor):
        if velocity_tensor.dim() != 2 or velocity_tensor.shape != _VELOCITY_SHAPE:
            raise ValueError(
                f"velocity_tensor must have shape {_VELOCITY_SHAPE}, got {tuple(velocity_tensor.shape)}"
            )
        if velocity_tensor.device.type == 'cuda':
            device_id = velocity_tensor.device.index if velocity_tensor.device.index is not None else 0
            cp.cuda.Device(device_id).use()

        ctx.save_for_backward(velocity_tensor)

        velocity_2d = velocity_tensor
        if velocity_2d.device.type == 'cuda':
            velocity_2d_double = velocity_2d.detach().double()
            velocity_cp = cp.from_dlpack(to_dlpack(velocity_2d_double))
        else:
            velocity_np = velocity_2d.detach().cpu().numpy()
            velocity_cp = cp.asarray(velocity_np, dtype=cp.float64)

        min_vel = cp.min(velocity_cp)
        v_flat = velocity_cp.flatten()
        min_vel_array = cp.array([min_vel], dtype=cp.float64)
        v_vector = cp.concatenate([v_flat, min_vel_array])
        v_vector_gpu = v_vector.reshape(-1, 1)

        result, _, _ = vel_to_seis(v_vector_gpu)
        seismic_data = result.reshape(5, 1000, 70)

        if velocity_tensor.device.type == 'cuda':
            result_tensor = from_dlpack(seismic_data.toDlpack())
        else:
            result_tensor = torch.from_numpy(cp.asnumpy(seismic_data)).to(velocity_tensor.device)

        return result_tensor.float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if grad_output.shape != (5, 1000, 70):
            raise ValueError(
                f"grad_output must have shape (5, 1000, 70), got {tuple(grad_output.shape)}"
            )
        if grad_output.device.type == 'cuda':
            device_id = grad_output.device.index if grad_output.device.index is not None else 0
            cp.cuda.Device(device_id).use()

        velocity_tensor = ctx.saved_tensors[0]
        velocity_2d = velocity_tensor
        grad_2d = grad_output

        if grad_2d.device.type == 'cuda':
            grad_2d_double = grad_2d.detach().double()
            grad_cp = cp.from_dlpack(to_dlpack(grad_2d_double))
        else:
            grad_np = grad_2d.detach().cpu().numpy()
            grad_cp = cp.asarray(grad_np, dtype=cp.float64)

        grad_vector = grad_cp.flatten().reshape(-1, 1)

        if velocity_2d.device.type == 'cuda':
            velocity_2d_double = velocity_2d.detach().double()
            velocity_cp = cp.from_dlpack(to_dlpack(velocity_2d_double))
        else:
            velocity_np = velocity_2d.detach().cpu().numpy()
            velocity_cp = cp.asarray(velocity_np, dtype=cp.float64)
        min_vel = cp.min(velocity_cp)
        v_flat = velocity_cp.flatten()
        min_vel_array = cp.array([min_vel], dtype=cp.float64)
        v_vector = cp.concatenate([v_flat, min_vel_array])
        v_vector_gpu = v_vector.reshape(-1, 1)

        _, _, adjoint_grad = vel_to_seis(
            v_vector_gpu,
            vec_adjoint=grad_vector,
            adjoint_on_residual=False,
        )

        velocity_grad = adjoint_grad[:-1, 0]
        velocity_grad_reshaped = velocity_grad.reshape(velocity_2d.shape)

        if velocity_tensor.device.type == 'cuda':
            velocity_grad_tensor = from_dlpack(velocity_grad_reshaped.toDlpack())
        else:
            velocity_grad_tensor = torch.from_numpy(cp.asnumpy(velocity_grad_reshaped)).to(velocity_tensor.device)

        velocity_grad_tensor = velocity_grad_tensor.to(velocity_tensor.dtype)
        return velocity_grad_tensor


def seismic_master_forward_modeling(velocity_tensor: torch.Tensor) -> torch.Tensor:
    """
    CuPy forward + PyTorch autograd. Velocity ``(70, 70)``; seismograms ``(5, 1000, 70)``.
    Wavelet, grid, and acquisition are fixed by this module's globals.
    """
    return SeismicMasterForwardModelingFunction.apply(velocity_tensor)


def torch_forward_modeling_gpu(velocity_tensor: torch.Tensor) -> torch.Tensor:
    """Alias of ``seismic_master_forward_modeling``."""
    return seismic_master_forward_modeling(velocity_tensor)
