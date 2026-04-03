# ============================================================
#  rank_sweep_gpt.py
#  GPT rank sweep — structured model D_ij = u_i · V[j,:]
#  where u_i = (φ_{c(i)}^T s_{config(i)}) is a per-row K-vector
#  with normalisation constraint u_i[0] = 1.
#  φ (16 phase matrices) and s (per-config state) are implicit;
#  u is fitted freely per row with the pinned-first-entry constraint.
#  Sharing φ across n_open cases is a future joint-fit extension.
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
_K_RANGE_GPT   = range(1, 21)
_N_REST_GPT    = 4
_ALS_MAX_ITER  = 500
_ALS_TOL       = 1e-7

_INSET_K_START = {1: 1, 2: 3, 3: 7, 4: 12}


# ── GPT ALS fit ────────────────────────────────────────────────────────────
def gpt_als_fit(data, sigma, K,
                train_mask=None, n_restarts=_N_REST_GPT,
                max_iter=_ALS_MAX_ITER, tol=_ALS_TOL,
                reg=ALS_REG, rng=None):
    """
    GPT rank-K fit: D_ij = u_i · V[j,:]
    Normalisation: u_i[0] = 1 for all rows i.

    u_i absorbs the slit-config state and phase transformation
    for row i; V[j,:] = X_j·e is the pixel factor.

    Parameters
    ----------
    data      : (n_rows, n_pix)
    sigma     : (n_rows, n_pix) uncertainties
    K         : GPT rank
    train_mask: (n_rows, n_pix) bool, True = training entry

    Returns
    -------
    u, V, train_chi2, test_chi2
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
    Us, ss, Vts = np.linalg.svd(data, full_matrices=False)
    u_init = Us[:, :K] * ss[:K]    # (n_rows, K)
    V_init = Vts[:K, :].T          # (n_pix,  K)
    u_init[:, 0] = 1.0             # enforce normalisation

    best_u, best_V  = None, None
    best_train_chi2 = np.inf
    best_test_chi2  = np.inf
    resid2          = np.zeros_like(data)

    for restart in range(n_restarts):
        if restart == 0:
            u = u_init.copy()
            V = V_init.copy()
        else:
            scale = 0.1 * max(float(u_init.std()), 1e-6)
            u = u_init + rng.standard_normal(u_init.shape) * scale
            V = V_init + rng.standard_normal(V_init.shape) * scale
        u[:, 0] = 1.0

        prev_chi2 = np.inf
        for _ in range(max_iter):

            # ── Update V ──────────────────────────────────────────────────
            # A_V[j,k,l] = Σ_i w[i,j] u[i,k] u[i,l] + reg I
            A_V = np.einsum('ij,ik,il->jkl', weights, u, u) + I_K[None]
            b_V = np.einsum('ij,ij,ik->jk',  weights, data, u)
            V   = np.linalg.solve(A_V, b_V[..., None])[..., 0]  # (n_pix, K)

            # ── Update u (per row, pin first component) ─────────────────
            V0 = V[:, 0]   # (n_pix,) shared baseline
            Vr = V[:, 1:]  # (n_pix, K-1)

            for i in range(n_rows):
                u[i, 0] = 1.0
                if K == 1:
                    continue
                w_i   = weights[i, :]          # (n_pix,)
                rhs_i = data[i, :] - V0        # (n_pix,)
                # A_u[k,l] = Σ_j w[i,j] Vr[j,k] Vr[j,l] + reg I
                A_u = np.einsum('j,jk,jl->kl', w_i, Vr, Vr) + I_Km1
                b_u = np.einsum('j,j,jk->k',   w_i, rhs_i, Vr)
                u[i, 1:] = np.linalg.solve(A_u, b_u)

            # ── Convergence ───────────────────────────────────────────────
            pred   = u @ V.T
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


# ── Single CV fold ─────────────────────────────────────────────────────────
def _fit_one_gpt(data_mat, sigma_mat, fold_ids, K, f, reg):
    train_mask = (fold_ids != f)
    for attempt in range(4):
        try:
            _, _, tr, te = gpt_als_fit(
                data_mat, sigma_mat, K=K,
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
        delayed(_fit_one_gpt)(data_mat, sigma_mat, fold_ids, K, f, sweep_reg)
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


# ── Helper: row-normalise for display ───────────────────────────────────────
def _row_norm(mat):
    m = mat.astype(float).copy()
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (m - lo) / rng


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
        im = ax_img.imshow(_row_norm(mat), aspect='auto', origin='lower',
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
        fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.03,
                     label='row-norm. intensity')

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
        axins = inset_axes(ax, width='48%', height='45%',
                           loc='upper right', borderpad=0.8)
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


# ── Entry point ─────────────────────────────────────────────────────────────
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
            f'u_i[0]=1 normalisation  |  '
            f'Dashed: χ²/pt=1  |  Green: expected rank'
        ),
        panel_prefix='Theory (noisy)',
    )
