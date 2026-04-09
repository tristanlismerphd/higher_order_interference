# rank_sweep_structured.py
# Structured GPT rank sweep using physics-derived channel matrices.
#
# Model:  D[s,x] = (T_s^K @ c) · v_x
#
#   T_s^K  = top-left K×K block of the real Liouville channel matrix M_s
#            M_s is KNOWN from experiment: slit mask × phase pattern
#            T_s^K[:K,:K] corresponds to the first K Hilbert-Schmidt modes:
#              K=1–3  : fewer than 4 diagonal modes  (extreme underfit)
#              K=4    : diagonal (populations) only   (classical incoherent)
#              K=10   : + real coherences (cos terms)
#              K=16   : full quantum mechanics
#              K=15   : QM minus one mode (trace-1 → expected test minimum)
#
#   c ∈ R^K  : density-matrix components (SHARED across ALL rows/configs)
#   V ∈ R^{n_pix × K}: pixel effects    (SHARED across ALL rows/configs)
#
# Only c and V are optimised via bilinear ALS; T_s is never a free variable.

import numpy as np
import time
from itertools import product as _prod
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError, solve
from foundations import RANDOM_SEED, ALS_REG
from rank_sweep_gpt import _N_FOLDS, _poisson_sigma, _resample_cols, N_PX_SWEEP

# ── Constants ──────────────────────────────────────────────────────────────────
N_SLITS = 4
N2      = N_SLITS ** 2   # 16 = dimension of full density-matrix space
_K_RANGE_STRUCT = range(1, N2 + 1)   # K = 1 … 16

_PHASE_PATTERNS_STRUCT = list(_prod([0.0, np.pi / 4, np.pi / 2], repeat=N_SLITS))

_SLIT_CONFIGS_STRUCT = [
    (0,  'XXXX', 0), (1,  'XXOX', 1), (2,  'XXXO', 1),
    (3,  'XXOO', 2), (4,  'OXXX', 1), (5,  'OXOX', 2),
    (6,  'OXXO', 2), (7,  'OXOO', 3), (8,  'XOXX', 1),
    (9,  'XOOX', 2), (10, 'XOXO', 2), (11, 'XOOO', 3),
    (12, 'OOXX', 2), (13, 'OOOX', 3), (14, 'OOXO', 3),
    (15, 'OOOO', 4),
]
_SLIT_MASK_STRUCT = {
    idx: np.array([0.0 if c == 'X' else 1.0 for c in pat])
    for idx, pat, _ in _SLIT_CONFIGS_STRUCT
}


# ── Hilbert-Schmidt basis ──────────────────────────────────────────────────────
def _build_hs_basis():
    """
    Real orthonormal HS basis for 4×4 Hermitian matrices (16 elements).
    Order:
      0–3  : diagonal  |k><k|
      4–9  : real off-diagonal  (|k><l|+|l><k|)/√2   k<l
      10–15: imaginary off-diagonal  i(|l><k|-|k><l|)/√2  k<l
    """
    n, basis = N_SLITS, []
    for k in range(n):
        m = np.zeros((n, n), dtype=complex); m[k, k] = 1.0
        basis.append(m)
    for k in range(n):
        for l in range(k + 1, n):
            m = np.zeros((n, n), dtype=complex)
            m[k, l] = m[l, k] = 1.0 / np.sqrt(2)
            basis.append(m)
    for k in range(n):
        for l in range(k + 1, n):
            m = np.zeros((n, n), dtype=complex)
            m[k, l] = -1j / np.sqrt(2);  m[l, k] = 1j / np.sqrt(2)
            basis.append(m)
    return basis   # list of 16 (4×4) complex Hermitian matrices

_HS_BASIS = _build_hs_basis()


# ── Channel matrix ─────────────────────────────────────────────────────────────
def _channel_matrix(A):
    """
    Real 16×16 Liouville channel matrix for  ρ → A ρ A†.
    A : (4,) complex amplitude vector  (0 for closed slits, e^{iφ} for open).
    M[α,β] = Tr(λ_α · diag(A) · λ_β · diag(A†))
    """
    M = np.zeros((N2, N2))
    for alpha, Ba in enumerate(_HS_BASIS):
        for beta, Bb in enumerate(_HS_BASIS):
            ABbAd = A[:, None] * Bb * np.conj(A)[None, :]
            M[alpha, beta] = np.real(np.sum(np.conj(Ba) * ABbAd))
    return M


def build_all_channel_matrices(row_configs):
    """
    Precompute M_s for every row.
    row_configs : list of (slit_idx, phase_idx) tuples in row order.
    Returns     : (n_rows, N2, N2) real float64 array.
    """
    print(f'  Computing {len(row_configs)} Liouville channel matrices...')
    t0 = time.time()
    Ms = []
    for slit_idx, phase_idx in row_configs:
        mask = _SLIT_MASK_STRUCT[slit_idx]
        phi  = np.array(_PHASE_PATTERNS_STRUCT[phase_idx])
        A    = mask * np.exp(1j * phi)
        Ms.append(_channel_matrix(A))
    M_all = np.array(Ms, dtype=float)
    print(f'  Done in {time.time() - t0:.1f}s')
    return M_all   # (n_rows, 16, 16)


# ── ALS for structured model ───────────────────────────────────────────────────
def _als_structured(data, sigma, T_mats, train_mask,
                    n_restarts=4, max_iter=300, tol=1e-6, reg=ALS_REG, rng=None):
    """
    Bilinear ALS:  D[s,x] ≈ (T_mats[s] @ c) · V[x]
    T_mats : (n_rows, K, K)  — fixed, derived from physics
    c      : (K,)             — optimised, shared across all rows
    V      : (n_pix, K)       — optimised, shared across all configs
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_rows, n_pix = data.shape
    K = T_mats.shape[-1]
    test_mask = ~train_mask
    w    = train_mask.astype(float) / np.maximum(sigma, 1e-12) ** 2
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    I_K  = reg * np.eye(K)

    best_c, best_V          = None, None
    best_train_chi2         = np.inf
    best_test_chi2          = np.inf
    resid2                  = np.zeros_like(data)

    for restart in range(n_restarts):
        c = rng.standard_normal(K) * 0.1
        if K >= 1:
            c[0] = 1.0
        V = rng.standard_normal((n_pix, K)) * 0.1

        prev_chi2 = np.inf

        for _ in range(max_iter):
            # ---- update c (fix V) ----------------------------------------
            # Q[s,x,b] = (T_s^T @ v_x)[b] = Σ_a V[x,a] T[s,a,b]
            # Use vectorised einsum; Q is (n_rows, n_pix, K) — ~78 MB for K=16
            Q   = np.einsum('xa,sab->sxb', V, T_mats)
            wQ  = w[..., None] * Q                        # (n_rows, n_pix, K)
            A_c = np.einsum('sxb,sxc->bc', wQ, Q) + I_K  # (K, K)
            b_c = np.einsum('sxb,sx->b', wQ, data)        # (K,)
            try:
                c = solve(A_c, b_c)
            except LinAlgError:
                break

            # ---- update V (fix c) ----------------------------------------
            U   = np.einsum('sab,b->sa', T_mats, c)           # (n_rows, K)
            A_V = np.einsum('sx,sa,sb->xab', w, U, U) + I_K[None]  # (n_pix,K,K)
            b_V = np.einsum('sx,sx,sa->xa', w, data, U)            # (n_pix, K)
            try:
                V = solve(A_V, b_V[..., None])[..., 0]        # (n_pix, K)
            except LinAlgError:
                break

            pred       = np.einsum('sa,xa->sx', U, V)
            resid2     = (data - pred) ** 2
            train_chi2 = (resid2[train_mask] / sigma[train_mask] ** 2).sum() / max(n_tr, 1)
            if abs(prev_chi2 - train_chi2) / max(prev_chi2, 1e-12) < tol:
                break
            prev_chi2 = train_chi2

        test_chi2 = (resid2[test_mask] / sigma[test_mask] ** 2).sum() / max(n_te, 1)
        if train_chi2 < best_train_chi2:
            best_train_chi2, best_test_chi2 = train_chi2, test_chi2
            best_c, best_V = c.copy(), V.copy()

    return best_c, best_V, best_train_chi2, best_test_chi2


# ── One CV fold job ────────────────────────────────────────────────────────────
def _fit_one_structured(all_mat, sigma_mat, M_full, fold_ids, K, f, reg,
                         group_row_slices):
    T_mats     = M_full[:, :K, :K]   # (n_rows, K, K)
    train_mask = (fold_ids != f)
    for attempt in range(4):
        try:
            c, V, _, _ = _als_structured(
                all_mat, sigma_mat, T_mats, train_mask,
                reg=reg * (10 ** attempt),
                rng=np.random.default_rng(RANDOM_SEED + K * 100 + f),
            )
            U      = np.einsum('sab,b->sa', T_mats, c)
            pred   = np.einsum('sa,xa->sx', U, V)
            resid2 = (all_mat - pred) ** 2 / np.maximum(sigma_mat, 1e-12) ** 2

            group_chi2 = {}
            for key, sl in group_row_slices.items():
                r2, tr_m, te_m = resid2[sl], train_mask[sl], ~train_mask[sl]
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
        except (LinAlgError, Exception):
            pass

    dummy = {key: (1e6, 1e6) for key in list(group_row_slices.keys()) + ['all']}
    return K, f, dummy


# ── Main sweep ─────────────────────────────────────────────────────────────────
def run_gpt_rank_sweep_structured(mats_dict, N_eff_dict, row_configs_dict,
                                   label='', n_jobs=-1):
    """
    Structured GPT rank-K CV sweep (K = 1 … 16).

    Parameters
    ----------
    mats_dict        : {n_open: prob_mat}
    N_eff_dict       : {n_open: N_eff}  (already scaled)
    row_configs_dict : {n_open: [(slit_idx, phase_idx), ...]}

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

    # Stacked row configs for channel matrix computation
    all_row_configs = []
    for n in n_open_list:
        all_row_configs.extend(row_configs_dict[n])
    M_full = build_all_channel_matrices(all_row_configs)   # (n_rows, 16, 16)

    all_N_eff = float(np.mean([N_eff_dict[n] for n in n_open_list]))
    sweep_reg = max(ALS_REG, all_N_eff * 1e-6)

    rng_cv   = np.random.default_rng(RANDOM_SEED)
    flat_idx = rng_cv.permutation(n_rows * n_pix)
    fold_ids = (flat_idx % _N_FOLDS).reshape(n_rows, n_pix)

    ks = list(_K_RANGE_STRUCT)
    print(f'\n-- Structured GPT {label}  ({n_rows}x{n_pix},  '
          f'N_eff~{all_N_eff:.1e},  reg={sweep_reg:.2e}) --')
    print(f'   K = 1–{N2}, {_N_FOLDS}-fold CV, {len(ks)*_N_FOLDS} jobs...')
    t0 = time.time()

    flat = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_structured)(
            all_mat, sigma_mat, M_full, fold_ids, K, f, sweep_reg, group_row_slices
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
                parts.append(f'{"n="+str(key) if key!="all" else "all"} '
                              f'tr={tr:.3f}/te={te:.3f}')
        print('  ' + '  '.join(parts))

    return results
