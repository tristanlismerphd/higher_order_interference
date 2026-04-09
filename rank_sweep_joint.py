# rank_sweep_joint.py
# Joint GPT rank sweep: all slit configurations (n_open=1,2,3,4) are fitted
# simultaneously sharing ONE effects matrix V.
#
# Model: D_ij = u_i @ v_j^T,  u_i[0] = 1  (GPT normalization)
#   - V is shared across ALL n_open groups
#   - u_i is free per row (but constrained u_i[0]=1)
#
# This is the correct physical structure: pixel effects (V) are the same
# regardless of which slits are open; only the state (u_i) changes.
#
# Returns per-group chi2/pt so 1-slit, 2-slit, and all-configs panels
# all reflect the SAME jointly-fitted V.

import numpy as np
import time
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from foundations import RANDOM_SEED, ALS_REG
from rank_sweep_gpt import (
    _K_RANGE_GPT, _N_FOLDS, _poisson_sigma, _resample_cols, N_PX_SWEEP,
    gpt_als_fit,
)


def _fit_one_joint(all_mat, sigma_mat, fold_ids, K, f, reg, group_row_slices):
    """Fit joint rank-K model for CV fold f; return per-group chi2/pt."""
    train_mask = (fold_ids != f)
    for _ in range(4):
        try:
            u, V, _, _ = gpt_als_fit(
                all_mat, sigma_mat, K=K, train_mask=train_mask, reg=reg,
                rng=np.random.default_rng(RANDOM_SEED + K * 100 + f),
            )
            pred   = u @ V.T
            resid2 = (all_mat - pred) ** 2 / np.maximum(sigma_mat, 1e-12) ** 2

            group_chi2 = {}
            for key, sl in group_row_slices.items():
                r2   = resid2[sl, :]
                tr_m = train_mask[sl, :]
                te_m = ~tr_m
                n_tr = int(tr_m.sum())
                n_te = int(te_m.sum())
                group_chi2[key] = (
                    float(r2[tr_m].sum() / max(n_tr, 1)),
                    float(r2[te_m].sum() / max(n_te, 1)),
                )

            # 'all' group over the full joint matrix
            n_tr = int(train_mask.sum())
            n_te = int((~train_mask).sum())
            group_chi2['all'] = (
                float(resid2[train_mask].sum() / max(n_tr, 1)),
                float(resid2[~train_mask].sum() / max(n_te, 1)),
            )
            return K, f, group_chi2

        except LinAlgError:
            reg *= 10

    dummy = {key: (1e6, 1e6) for key in list(group_row_slices.keys()) + ['all']}
    return K, f, dummy


def run_gpt_rank_sweep_joint(mats_dict, N_eff_dict, label='', n_jobs=-1):
    """
    Joint GPT rank-K CV sweep: all n_open groups share one effects matrix V.

    Parameters
    ----------
    mats_dict  : dict  {n_open (int): prob_mat}  for n_open in {1,2,3,4}
    N_eff_dict : dict  {n_open (int): N_eff}     effective photon count per group

    Returns
    -------
    results : dict  keys are n_open ints plus 'all'.
              Each value: {K: {'train': list_of_floats, 'test': list_of_floats}}
    """
    n_open_list = sorted(k for k in mats_dict if isinstance(k, int))

    # Resample all matrices to the same number of pixels
    mats_r = {n: _resample_cols(mats_dict[n], N_PX_SWEEP) for n in n_open_list}

    # Row-slice for each group
    group_row_slices = {}
    row = 0
    for n in n_open_list:
        end = row + mats_r[n].shape[0]
        group_row_slices[n] = slice(row, end)
        row = end

    all_mat = np.vstack([mats_r[n] for n in n_open_list])
    n_rows, n_pix = all_mat.shape

    # Per-row sigma: each group uses its own N_eff
    sigma_mat = np.zeros_like(all_mat)
    for n in n_open_list:
        sl = group_row_slices[n]
        sigma_mat[sl, :] = _poisson_sigma(mats_r[n], N_eff_dict[n])

    all_N_eff = float(np.mean([N_eff_dict[n] for n in n_open_list]))
    sweep_reg = max(ALS_REG, all_N_eff * 1e-6)

    # Random pixel CV folds (same scheme as run_gpt_rank_sweep)
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

    # Aggregate
    all_keys = n_open_list + ['all']
    results = {key: {K: {'train': [], 'test': []} for K in ks} for key in all_keys}
    for K, f, group_chi2 in flat:
        for key in all_keys:
            tr, te = group_chi2[key]
            results[key][K]['train'].append(tr)
            results[key][K]['test'].append(te)

    # Print summary
    for K in ks:
        parts = [f'K={K:>2}']
        for key in [1, 2, 'all']:
            if key not in results:
                continue
            tr = np.mean(results[key][K]['train'])
            te = np.mean(results[key][K]['test'])
            lbl = f'n={key}' if key != 'all' else 'all'
            parts.append(f'{lbl} tr={tr:.3f}/te={te:.3f}')
        print('  ' + '  '.join(parts))

    return results
