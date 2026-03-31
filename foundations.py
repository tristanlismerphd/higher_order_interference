# ============================================================
#  Foundations — imports, constants, helpers, ALS fit
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import os, re, time
from itertools import product as _prod

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
SIGMA_FLOOR = 0.01

# Simulation beam parameters
BEAM_RADIUS = 0.8
KX_LIST     = [-20, -10, 10, 20]
SLIT_X      = np.array([-0.6, -0.2, 0.2, 0.6])
NUM_PIXELS  = 150
x_grid      = np.linspace(-3 * BEAM_RADIUS, 3 * BEAM_RADIUS, NUM_PIXELS)

# Phase patterns: {0, π/2}^4 = 16 settings
phase_patterns = list(_prod([0.0, np.pi / 2], repeat=4))

# ── Helpers ───────────────────────────────────────────────────────────────────
def _row_minmax(mat):
    """Row-wise min-max normalisation → values in [0, 1]."""
    out  = mat.astype(float).copy()
    mins = out.min(axis=1, keepdims=True)
    rng  = out.max(axis=1, keepdims=True) - mins
    rng[rng == 0] = 1.0
    return (out - mins) / rng

# ── ALS rank-K matrix factorisation ──────────────────────────────────────────
ALS_MAX_ITER = 500
ALS_TOL      = 1e-7
ALS_REG      = 1e-6
N_RESTARTS   = 8

def als_fit(data, sigma, train_mask, K,
            n_restarts=N_RESTARTS, max_iter=ALS_MAX_ITER,
            tol=ALS_TOL, reg=ALS_REG, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_s, n_pix = data.shape
    K          = min(K, n_s, n_pix)
    weights    = train_mask.astype(float) / sigma**2
    test_mask  = ~train_mask
    wdata      = weights * data
    I_K        = reg * np.eye(K)
    U_svd, s_svd, Vt_svd = np.linalg.svd(data, full_matrices=False)
    U_init = U_svd[:, :K] * s_svd[:K]
    W_init = Vt_svd[:K, :].T
    best_U, best_W  = None, None
    best_train_chi2 = np.inf
    n_train = train_mask.sum()
    n_test  = test_mask.sum()
    for restart in range(n_restarts):
        if restart == 0:
            U, W = U_init.copy(), W_init.copy()
        else:
            scale = 0.1 * max(U_init.std(), 1e-6)
            U = U_init + rng.standard_normal(U_init.shape) * scale
            W = W_init + rng.standard_normal(W_init.shape) * scale
        prev_chi2 = np.inf
        for it in range(max_iter):
            A_U = np.einsum('sx,xk,xl->skl', weights, W, W) + I_K
            b_U = wdata @ W
            U   = np.linalg.solve(A_U, b_U[..., None])[..., 0]
            A_W = np.einsum('sx,sk,sl->xkl', weights, U, U) + I_K
            b_W = wdata.T @ U
            W   = np.linalg.solve(A_W, b_W[..., None])[..., 0]
            resid2          = (data - U @ W.T)**2
            train_chi2_iter = (resid2[train_mask] / sigma[train_mask]**2).sum() / n_train
            if abs(prev_chi2 - train_chi2_iter) / max(prev_chi2, 1e-12) < tol:
                break
            prev_chi2 = train_chi2_iter
        test_chi2_restart = (resid2[test_mask] / sigma[test_mask]**2).sum() / n_test
        if train_chi2_iter < best_train_chi2:
            best_train_chi2 = train_chi2_iter
            best_test_chi2  = test_chi2_restart
            best_U, best_W  = U.copy(), W.copy()
    return best_U, best_W, best_train_chi2, best_test_chi2


if __name__ == '__main__':
    print('Foundations loaded.')
    print(f'  phase_patterns : {len(phase_patterns)} settings  ({{0, π/2}}^4)')
    print(f'  x_grid         : {NUM_PIXELS} pts  [{x_grid[0]:.2f}, {x_grid[-1]:.2f}]')
    print(f'  SLIT_X         : {SLIT_X}')
    print(f'  als_fit        : ready')
