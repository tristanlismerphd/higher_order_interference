# ============================================================
#  rank_sweep_gpt.py
#  GPT rank sweep — structured model D_ij = (s·φ_c)_i · (X·e)_j
#  φ_c is shared across all rows with the same phase pattern.
#  There are 16 phase patterns: {0, π/2}^4.
#  Normalisation constraint: (s·φ_c)[0] = 1 for all c.
#  (K=1-20, 10-fold CV, parallelised)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.linalg import LinAlgError
from foundations import RANDOM_SEED, ALS_REG
from data import build_simulation_data, build_theory_data
from rank_sweep_noisy import _poisson_sigma, _resample_cols, _N_FOLDS, N_PX_SWEEP

# ── Sweep parameters ───────────────────────────────────────────────────────
_K_RANGE_GPT   = range(1, 21)   # rank 1 – 20
_N_REST_GPT    = 4
_ALS_MAX_ITER  = 500
_ALS_TOL       = 1e-7
N_PHASES       = 16             # {0, π/2}^4

_INSET_K_START = {1: 1, 2: 3, 3: 7, 4: 12}


# ── GPT ALS fit ────────────────────────────────────────────────────────────
def gpt_als_fit(data, sigma, phase_idx, K,
                train_mask=None, n_restarts=_N_REST_GPT,
                max_iter=_ALS_MAX_ITER, tol=_ALS_TOL,
                reg=ALS_REG, rng=None):
    """
    GPT rank-K matrix factorisation: D_ij = u_{c(i)} · V[j, :]
    where c(i) = phase_idx[i] in {0,...,15} and u_c[0] = 1.

    u_c  = s·φ_c   — one normalised K-vector per phase pattern
    V[j] = X_j·e   — one K-vector per pixel (free)

    Parameters
    ----------
    data      : (n_rows, n_pix) array
    sigma     : (n_rows, n_pix) uncertainty array
    phase_idx : (n_rows,) int array — phase-pattern index for each row
    K         : GPT rank
    train_mask: (n_rows, n_pix) bool — True = training entry

    Returns
    -------
    u          : (16, K) — one normalised vector per phase pattern
    V          : (n_pix, K) — pixel factor matrix
    train_chi2 : float
    test_chi2  : float
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_rows, n_pix = data.shape
    K = max(1, min(K, n_rows, n_pix))

    if train_mask is None:
        train_mask = np.ones((n_rows, n_pix), dtype=bool)
    test_mask = ~train_mask

    weights = train_mask.astype(float) / np.maximum(sigma, 1e-12) ** 2
    n_train = int(train_mask.sum())
    n_test  = int(test_mask.sum())

    I_K   = reg * np.eye(K)
    I_Km1 = reg * np.eye(K - 1) if K > 1 else np.zeros((0, 0))

    # SVD initialisation
    U_svd, s_svd, Vt_svd = np.linalg.svd(data, full_matrices=False)
    U_init = U_svd[:, :K] * s_svd[:K]   # (n_rows, K)
    V_init = Vt_svd[:K, :].T             # (n_pix, K)

    # Average SVD rows per phase pattern to initialise u
    u_init = np.zeros((N_PHASES, K))
    for c in range(N_PHASES):
        mask_c = (phase_idx == c)
        u_init[c] = U_init[mask_c].mean(axis=0) if mask_c.any() else rng.standard_normal(K) * 0.1
    u_init[:, 0] = 1.0   # enforce normalisation

    best_u, best_V  = None, None
    best_train_chi2 = np.inf
    best_test_chi2  = np.inf
    resid2          = np.zeros_like(data)

    for restart in range(n_restarts):
        if restart == 0:
            u = u_init.copy()
            V = V_init.copy()
        else:
            scale = 0.1 * max(float(U_init.std()), 1e-6)
            u = u_init + rng.standard_normal(u_init.shape) * scale
            V = V_init + rng.standard_normal(V_init.shape) * scale
        u[:, 0] = 1.0

        prev_chi2 = np.inf
        for _ in range(max_iter):

            # ── Update V (all pixels simultaneously) ──────────────────────
            U_row = u[phase_idx]   # (n_rows, K)
            # A_V[j,k,l] = sum_i w[i,j] * U_row[i,k] * U_row[i,l] + reg*I
            A_V = np.einsum('ij,ik,il->jkl', weights, U_row, U_row) + I_K[None]
            # b_V[j,k] = sum_i w[i,j] * data[i,j] * U_row[i,k]
            b_V = np.einsum('ij,ij,ik->jk',  weights, data,  U_row)
            # solve A_V[j] @ V[j] = b_V[j] for each pixel j
            V = np.linalg.solve(A_V, b_V[..., None])[..., 0]   # (n_pix, K)

            # ── Update u_c (one phase pattern at a time) ──────────────────
            # Split: u_c = [1, u_c_free], V = [V0, Vr]
            # Residual after pinned component: rhs[i,j] = data[i,j] - V[j,0]
            V0 = V[:, 0]    # (n_pix,)
            Vr = V[:, 1:]   # (n_pix, K-1)

            for c in range(N_PHASES):
                rows_c = np.where(phase_idx == c)[0]
                if not rows_c.size:
                    continue
                if K == 1:
                    u[c, 0] = 1.0
                    continue
                w_c   = weights[rows_c, :]              # (n_c, n_pix)
                rhs   = data[rows_c, :] - V0[None, :]  # (n_c, n_pix)
                w_sum = w_c.sum(axis=0)                 # (n_pix,)
                # A_u[k,l] = sum_j w_sum[j] * Vr[j,k] * Vr[j,l] + reg*I
                A_u = np.einsum('j,jk,jl->kl', w_sum, Vr, Vr) + I_Km1
                # b_u[k] = sum_{i,j} w_c[i,j] * rhs[i,j] * Vr[j,k]
                b_u = np.einsum('ij,ij,jk->k', w_c, rhs, Vr)
                u[c, 0]  = 1.0
                u[c, 1:] = np.linalg.solve(A_u, b_u)

            # ── Convergence check ─────────────────────────────────────────
            U_row  = u[phase_idx]
            pred   = U_row @ V.T          # (n_rows, n_pix)
            resid2 = (data - pred) ** 2
            train_chi2 = (
                (resid2[train_mask] / sigma[train_mask] ** 2).sum()
                / max(n_train, 1)
            )
            if abs(prev_chi2 - train_chi2) / max(prev_chi2, 1e-12) < tol:
                break
            prev_chi2 = train_chi2

        test_chi2 = (
            (resid2[test_mask] / sigma[test_mask] ** 2).sum()
            / max(n_test, 1)
        )
        if train_chi2 < best_train_chi2:
            best_train_chi2 = train_chi2
            best_test_chi2  = test_chi2
            best_u = u.copy()
            best_V = V.copy()

    return best_u, best_V, best_train_chi2, best_test_chi2


# ── Phase index helper ─────────────────────────────────────────────────────
def _get_phase_idx(n_rows):
    """Phase pattern is the inner loop (16 patterns), so idx = row % 16."""
    return np.tile(np.arange(N_PHASES), -(-n_rows // N_PHASES))[:n_rows]


# ── Single CV fold ─────────────────────────────────────────────────────────
def _fit_one_gpt(data_mat, sigma_mat, phase_idx, fold_ids, K, f, reg):
    train_mask = (fold_ids != f)
    for attempt in range(4):
        try:
            _, _, tr, te = gpt_als_fit(
                data_mat, sigma_mat, phase_idx, K=K,
                train_mask=train_mask, reg=reg,
                rng=np.random.default_rng(RANDOM_SEED + K * 100 + f),
            )
            return K, f, tr, te
        except LinAlgError:
            reg *= 10
    return K, f, 1e6, 1e6


# ── Rank sweep ─────────────────────────────────────────────────────────────
def run_gpt_rank_sweep(data_mat, N_eff, label='', n_jobs=-1):
    """10-fold CV GPT rank sweep (K = 1–20)."""
    data_mat      = _resample_cols(data_mat, N_PX_SWEEP)
    n_rows, n_pix = data_mat.shape
    sigma_mat     = _poisson_sigma(data_mat, N_eff)
    phase_idx     = _get_phase_idx(n_rows)
    sweep_reg     = max(ALS_REG, N_eff * 1e-6)

    rng_cv   = np.random.default_rng(RANDOM_SEED)
    perm     = rng_cv.permutation(n_rows * n_pix)
    fold_ids = np.empty(n_rows * n_pix, dtype=int)
    for f in range(_N_FOLDS):
        fold_ids[perm[f::_N_FOLDS]] = f
    fold_ids = fold_ids.reshape(n_rows, n_pix)

    ks = list(_K_RANGE_GPT)
    print(f'\n── GPT {label}  ({n_rows}×{n_pix},  N_eff={N_eff:.1f},  reg={sweep_reg:.2e}) ──')
    print(f'   Running {len(ks) * _N_FOLDS} jobs in parallel (n_jobs={n_jobs})...')

    t0   = time.time()
    jobs = [(K, f) for K in ks for f in range(_N_FOLDS)]
    flat = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_gpt)(data_mat, sigma_mat, phase_idx, fold_ids, K, f, sweep_reg)
        for K, f in jobs
    )
    print(f'   Done in {time.time() - t0:.1f}s')

    results = {K: {'train': [], 'test': []} for K in ks}
    for K, f, tr, te in flat:
        results[K]['train'].append(tr)
        results[K]['test'].append(te)

    for K in ks:
        print(f'  K={K:>2}  train={np.mean(results[K]["train"]):.4f}  '
              f'test={np.mean(results[K]["test"]):.4f}')
    return results


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_gpt_sweep(cv_dict, mat_dict, mats_N, suptitle, panel_prefix):
    ks    = list(_K_RANGE_GPT)
    x_pos = np.arange(len(ks))
    width = 0.38

    fig, axes = plt.subplots(2, 4, figsize=(28, 10),
                             gridspec_kw={'height_ratios': [1, 2]})
    fig.subplots_adjust(wspace=0.38, hspace=0.35)

    for col, n_open in enumerate([1, 2, 3, 4]):
        mat   = mat_dict[n_open]
        N_eff = mats_N[n_open]
        k_ins = _INSET_K_START[n_open]

        ax_img = axes[0, col]
        im = ax_img.imshow(mat, aspect='auto', origin='lower',
                           cmap='magma', vmin=0, vmax=1)
        ax_img.set_title(
            f'{panel_prefix}  |  {n_open}-slit\n'
            f'{mat.shape[0]} settings × {mat.shape[1]} px  '
            f'(expected rank={n_open**2})',
            fontsize=10,
        )
        ax_img.set_xlabel('pixel index', fontsize=9)
        ax_img.set_ylabel('setting index', fontsize=9)
        ax_img.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.03, label='row-norm. intensity')

        ax = axes[1, col]
        tr_means = [np.mean(cv_dict[n_open][K]['train']) for K in ks]
        tr_stds  = [np.std (cv_dict[n_open][K]['train']) for K in ks]
        te_means = [np.mean(cv_dict[n_open][K]['test'])  for K in ks]
        te_stds  = [np.std (cv_dict[n_open][K]['test'])  for K in ks]

        ax.bar(x_pos - width/2, tr_means, width, yerr=tr_stds, capsize=3,
               color='steelblue', alpha=0.85, ecolor='navy',    label='Train')
        ax.bar(x_pos + width/2, te_means, width, yerr=te_stds, capsize=3,
               color='tomato',    alpha=0.85, ecolor='darkred', label='Test')
        ax.axhline(1.0, color='gray',  linestyle='--', linewidth=1)
        ax.axvline(n_open**2 - 1, color='green', linestyle=':', linewidth=1.2,
                   label=f'Expected rank={n_open**2}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7, rotation=45)
        ax.set_xlabel('GPT rank K', fontsize=10)
        ax.set_ylabel('χ²/pt', fontsize=10)
        ax.set_title(f'N_eff={N_eff:.0f}', fontsize=10)
        ax.legend(fontsize=7)

        inset_idx  = [i for i, k in enumerate(ks) if k >= k_ins]
        inset_ks   = [ks[i] for i in inset_idx]
        inset_xpos = np.arange(len(inset_ks))
        axins = inset_axes(ax, width='48%', height='45%', loc='upper right', borderpad=0.8)
        axins.bar(inset_xpos - width/2, [tr_means[i] for i in inset_idx], width,
                  yerr=[tr_stds[i] for i in inset_idx], capsize=2,
                  color='steelblue', alpha=0.85, ecolor='navy')
        axins.bar(inset_xpos + width/2, [te_means[i] for i in inset_idx], width,
                  yerr=[te_stds[i] for i in inset_idx], capsize=2,
                  color='tomato', alpha=0.85, ecolor='darkred')
        axins.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        axins.set_xticks(inset_xpos)
        axins.set_xticklabels([str(k) for k in inset_ks], fontsize=6, rotation=45)
        axins.tick_params(axis='y', labelsize=6)
        axins.set_xlabel('K', fontsize=7)
        axins.set_title(f'K ≥ {k_ins}', fontsize=7, pad=2)
        inset_vals = (
            [tr_means[i] - tr_stds[i] for i in inset_idx] +
            [te_means[i] - te_stds[i] for i in inset_idx] +
            [tr_means[i] + tr_stds[i] for i in inset_idx] +
            [te_means[i] + te_stds[i] for i in inset_idx]
        )
        lo, hi = min(inset_vals), max(inset_vals)
        pad_v  = max((hi - lo) * 0.15, 0.05)
        axins.set_ylim(lo - pad_v, hi + pad_v)

    plt.suptitle(suptitle, fontsize=13)
    plt.show()


# ── Entry point: noisy theory data ────────────────────────────────────────
if __name__ == '__main__':
    _, _, mats_N             = build_simulation_data()
    theory_mats, _, th_N_eff = build_theory_data(add_noise=True)
    theory_mats_N = {n: th_N_eff for n in [1, 2, 3, 4]}

    gpt_cv = {}
    for n_open in [1, 2, 3, 4]:
        gpt_cv[n_open] = run_gpt_rank_sweep(
            theory_mats[n_open], N_eff=th_N_eff,
            label=f'Theory (noisy)  {n_open}-slit',
        )

    plot_gpt_sweep(
        gpt_cv, theory_mats, theory_mats_N,
        suptitle=(
            f'GPT rank sweep — s·φ_c·X·e model  |  '
            f'Theory + Poisson noise  |  {_N_FOLDS}-fold CV  |  N_eff={th_N_eff}\n'
            f'phases: {{0, π/2}}^4  |  Normalisation: (s·φ_c)[0]=1  |  '
            f'Dashed: χ²/pt=1  |  Green: expected rank'
        ),
        panel_prefix='Theory (noisy)',
    )
