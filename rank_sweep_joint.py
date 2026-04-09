# rank_sweep_joint.py
# Joint GPT rank sweep matching the cluster/run_gpt.py model structure:
#
#   D[s,x] = (omega @ T[s]) · (X_all[x] @ e)
#
#   omega  : shared K-vector input state     (omega[0] = 1 normalisation)
#   T[s]   : free K×K per-row transformation  (one per slit+phase setting)
#   X_all[x]: free K×K per-pixel measurement  (shared structure)
#   e      : shared K-vector effect
#
# Since T[s] and X_all[x] are free, this reduces to a standard bilinear
# factorisation D ≈ U @ V^T with:
#   U[s] = omega @ T[s]  → free K-vector per row
#   V[x] = X_all[x] @ e → free K-vector per pixel
#
# Key difference from the old per-group runs: ALL n_open groups (1,2,3,4)
# are stacked into one matrix and fitted with a SINGLE shared V (pixel effects),
# so the 1-slit, 2-slit, and all-configs panels all reflect the same model.
# Per-group chi2/pt is computed after the fit so all three panels are reported.
#
# ALS replaces SLSQP for speed; no u[0]=1 per-row constraint (cluster convention
# is omega[0]=1 shared, not a per-row pin).

import numpy as np
import time
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError, solve
from foundations import RANDOM_SEED, ALS_REG
from rank_sweep_gpt import (
    _K_RANGE_GPT, _N_FOLDS, _poisson_sigma, _resample_cols, N_PX_SWEEP,
)


# ── Standard bilinear ALS (no per-row constraint) ──────────────────────────────
def _als_bilinear(data, sigma, K, train_mask,
                  n_restarts=4, max_iter=500, tol=1e-7, reg=ALS_REG, rng=None):
    """
    Weighted bilinear ALS:  D ≈ U @ V^T
    U : (n_rows, K)  — free, initialised from SVD
    V : (n_pix,  K)  — free, shared across all rows (= X_all @ e)
    No per-row normalisation constraint (matches cluster/run_gpt.py convention).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_rows, n_pix = data.shape
    test_mask = ~train_mask
    w    = train_mask.astype(float) / np.maximum(sigma, 1e-12) ** 2
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    I_K  = reg * np.eye(K)

    # SVD initialisation (same as cluster code's near-identity T, X init)
    Us, ss, Vts = np.linalg.svd(data, full_matrices=False)
    U_init = (Us[:, :K] * ss[:K]).astype(float)
    V_init = Vts[:K].T.astype(float)

    best_U, best_V          = None, None
    best_train_chi2         = np.inf
    best_test_chi2          = np.inf
    resid2                  = np.zeros_like(data)

    for restart in range(n_restarts):
        if restart == 0:
            U, V = U_init.copy(), V_init.copy()
        else:
            scale = 0.1 * max(float(U_init.std()), 1e-6)
            U = U_init + rng.standard_normal(U_init.shape) * scale
            V = V_init + rng.standard_normal(V_init.shape) * scale

        prev_chi2 = np.inf

        for _ in range(max_iter):
            # ---- update U (fix V) ----------------------------------------
            A_U = np.einsum('sx,xk,xl->skl', w, V, V) + I_K[None]  # (n_rows,K,K)
            b_U = np.einsum('sx,sx,xk->sk',  w, data, V)            # (n_rows,K)
            try:
                U = solve(A_U, b_U[..., None])[..., 0]
            except LinAlgError:
                break

            # ---- update V (fix U) ----------------------------------------
            A_V = np.einsum('sx,sk,sl->xkl', w, U, U) + I_K[None]  # (n_pix,K,K)
            b_V = np.einsum('sx,sx,sk->xk',  w, data, U)            # (n_pix,K)
            try:
                V = solve(A_V, b_V[..., None])[..., 0]
            except LinAlgError:
                break

            pred       = U @ V.T
            resid2     = (data - pred) ** 2
            train_chi2 = (resid2[train_mask] / sigma[train_mask] ** 2).sum() / max(n_tr, 1)
            if abs(prev_chi2 - train_chi2) / max(prev_chi2, 1e-12) < tol:
                break
            prev_chi2 = train_chi2

        test_chi2 = (resid2[test_mask] / sigma[test_mask] ** 2).sum() / max(n_te, 1)
        if train_chi2 < best_train_chi2:
            best_train_chi2, best_test_chi2 = train_chi2, test_chi2
            best_U, best_V = U.copy(), V.copy()

    return best_U, best_V, best_train_chi2, best_test_chi2


# ── One CV fold ────────────────────────────────────────────────────────────────
def _fit_one_joint(all_mat, sigma_mat, fold_ids, K, f, reg, group_row_slices):
    """Fit joint rank-K model for CV fold f; return per-group chi2/pt."""
    train_mask = (fold_ids != f)
    for attempt in range(4):
        try:
            U, V, _, _ = _als_bilinear(
                all_mat, sigma_mat, K=K, train_mask=train_mask,
                reg=reg * (10 ** attempt),
                rng=np.random.default_rng(RANDOM_SEED + K * 100 + f),
            )
            pred   = U @ V.T
            resid2 = (all_mat - pred) ** 2 / np.maximum(sigma_mat, 1e-12) ** 2

            group_chi2 = {}
            for key, sl in group_row_slices.items():
                r2   = resid2[sl, :]
                tr_m = train_mask[sl, :]
                te_m = ~tr_m
                group_chi2[key] = (
                    float(r2[tr_m].sum() / max(int(tr_m.sum()), 1)),
                    float(r2[te_m].sum() / max(int(te_m.sum()), 1)),
                )

            n_tr = int(train_mask.sum())
            n_te = int((~train_mask).sum())
            group_chi2['all'] = (
                float(resid2[train_mask].sum() / max(n_tr, 1)),
                float(resid2[~train_mask].sum() / max(n_te, 1)),
            )
            return K, f, group_chi2

        except LinAlgError:
            pass

    dummy = {key: (1e6, 1e6) for key in list(group_row_slices.keys()) + ['all']}
    return K, f, dummy


# ── Main sweep ─────────────────────────────────────────────────────────────────
def run_gpt_rank_sweep_joint(mats_dict, N_eff_dict, label='', n_jobs=-1):
    """
    Joint GPT rank-K CV sweep (K = 1–20).
    All n_open groups share one V (pixel effects = X_all @ e).
    Per-group chi2/pt reported for 1-slit, 2-slit, and all-configs panels.

    Parameters
    ----------
    mats_dict  : {n_open (int): prob_mat}
    N_eff_dict : {n_open (int): N_eff}

    Returns
    -------
    results : {n_open / 'all': {K: {'train': [...], 'test': [...]}}}
    """
    n_open_list = sorted(k for k in mats_dict if isinstance(k, int))
    mats_r      = {n: _resample_cols(mats_dict[n], N_PX_SWEEP) for n in n_open_list}

    group_row_slices, row = {}, 0
    for n in n_open_list:
        end = row + mats_r[n].shape[0]
        group_row_slices[n] = slice(row, end)
        row = end

    all_mat  = np.vstack([mats_r[n] for n in n_open_list])
    n_rows, n_pix = all_mat.shape

    sigma_mat = np.zeros_like(all_mat)
    for n in n_open_list:
        sl = group_row_slices[n]
        sigma_mat[sl] = _poisson_sigma(mats_r[n], N_eff_dict[n])

    all_N_eff = float(np.mean([N_eff_dict[n] for n in n_open_list]))
    sweep_reg = max(ALS_REG, all_N_eff * 1e-6)

    rng_cv   = np.random.default_rng(RANDOM_SEED)
    flat_idx = rng_cv.permutation(n_rows * n_pix)
    fold_ids = (flat_idx % _N_FOLDS).reshape(n_rows, n_pix)

    ks = list(_K_RANGE_GPT)
    print(f'\n-- GPT Joint {label}  ({n_rows}x{n_pix},  '
          f'N_eff~{all_N_eff:.1e},  reg={sweep_reg:.2e}) --')
    print(f'   Running {len(ks) * _N_FOLDS} jobs...')
    t0 = time.time()

    flat = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_joint)(
            all_mat, sigma_mat, fold_ids, K, f, sweep_reg, group_row_slices
        )
        for K in ks for f in range(_N_FOLDS)
    )
    print(f'   Done in {time.time() - t0:.1f}s')

    all_keys = n_open_list + ['all']
    results  = {key: {K: {'train': [], 'test': []} for K in ks} for key in all_keys}
    for K, f, group_chi2 in flat:
        for key in all_keys:
            tr, te = group_chi2[key]
            results[key][K]['train'].append(tr)
            results[key][K]['test'].append(te)

    for K in ks:
        parts = [f'K={K:>2}']
        for key in [1, 2, 'all']:
            if key in results:
                tr = np.mean(results[key][K]['train'])
                te = np.mean(results[key][K]['test'])
                parts.append(f'{"n="+str(key) if key != "all" else "all"} '
                              f'tr={tr:.3f}/te={te:.3f}')
        print('  ' + '  '.join(parts))

    return results
