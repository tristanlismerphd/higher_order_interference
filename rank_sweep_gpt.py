# ============================================================
#  rank_sweep_gpt.py
#  GPT rank sweep -- structured model D_ij = u_i * V[j,:]
#  where u_i is a per-row K-vector with u_i[0] = 1.
#  (K=1-20, 10-fold CV, parallelised)
#
#  Last change: added THEORY_CROP_THRESHOLD; plot theory matrices before sweep
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.linalg import LinAlgError
from foundations import RANDOM_SEED, ALS_REG
from data import build_theory_data

# -- Crop threshold: columns where mean intensity < this are dropped -------
THEORY_CROP_THRESHOLD = 0.05   # set to 0.0 to disable

# -- Sweep parameters --------------------------------------------------
_K_RANGE_GPT  = range(1, 21)
_N_REST_GPT   = 4
_ALS_MAX_ITER = 500
_ALS_TOL      = 1e-7
_INSET_KS       = list(range(12, 19))  # default inset range
_INSET_KS_1SLIT = list(range(2, 6))   # inset for 1-slit panel (K=2-5)
_TABLE_KS       = [14, 15, 16, 17, 18]
_TABLE_KS_1SLIT = [1, 2, 3, 4, 5]

# -- Noise / CV helpers (inlined from rank_sweep_noisy) ----------------
_N_FOLDS   = 10
N_PX_SWEEP = 500
_P_FLOOR   = 0.01


def _poisson_sigma(prob_mat, N_eff):
    return np.sqrt(np.maximum(prob_mat, _P_FLOOR)) / np.sqrt(N_eff)


def _resample_cols(mat, n_out):
    n_in = mat.shape[1]
    if n_in == n_out:
        return mat
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    return np.array([np.interp(x_out, x_in, row) for row in mat])


# -- Summary table helper ----------------------------------------------
def _add_rank_table(ax, cv_results, ks, table_ks=None):
    """Add a summary table below ax."""
    if table_ks is None:
        table_ks = _TABLE_KS
    col_labels = ['K', 'Train (mean\u00b1std)', 'Test (mean\u00b1std)']
    cell_text = []
    for K in table_ks:
        if K not in cv_results:
            continue
        tr = cv_results[K]['train']
        te = cv_results[K]['test']
        cell_text.append([
            str(K),
            f'{np.mean(tr):.4f} \u00b1 {np.std(tr):.4f}',
            f'{np.mean(te):.4f} \u00b1 {np.std(te):.4f}',
        ])
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='bottom',
        bbox=[0, -0.54, 1, 0.32],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)


# -- GPT ALS fit -------------------------------------------------------
def gpt_als_fit(data, sigma, K,
                train_mask=None, n_restarts=_N_REST_GPT,
                max_iter=_ALS_MAX_ITER, tol=_ALS_TOL,
                reg=ALS_REG, rng=None):
    """
    GPT rank-K fit: D_ij = u_i * V[j,:],  u_i[0] = 1.
    Returns u, V, train_chi2, test_chi2.
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

    Us, ss, Vts = np.linalg.svd(data, full_matrices=False)
    u_init = Us[:, :K] * ss[:K]
    V_init = Vts[:K, :].T
    u_init[:, 0] = 1.0

    best_u, best_V  = None, None
    best_train_chi2 = np.inf
    best_test_chi2  = np.inf
    resid2          = np.zeros_like(data)

    for restart in range(n_restarts):
        if restart == 0:
            u, V = u_init.copy(), V_init.copy()
        else:
            scale = 0.1 * max(float(u_init.std()), 1e-6)
            u = u_init + rng.standard_normal(u_init.shape) * scale
            V = V_init + rng.standard_normal(V_init.shape) * scale
        u[:, 0] = 1.0

        prev_chi2 = np.inf
        for _ in range(max_iter):
            A_V = np.einsum('ij,ik,il->jkl', weights, u, u) + I_K[None]
            b_V = np.einsum('ij,ij,ik->jk',  weights, data, u)
            V   = np.linalg.solve(A_V, b_V[..., None])[..., 0]

            V0, Vr = V[:, 0], V[:, 1:]
            for i in range(n_rows):
                u[i, 0] = 1.0
                if K == 1:
                    continue
                w_i   = weights[i, :]
                rhs_i = data[i, :] - V0
                A_u = np.einsum('j,jk,jl->kl', w_i, Vr, Vr) + I_Km1
                b_u = np.einsum('j,j,jk->k',   w_i, rhs_i, Vr)
                u[i, 1:] = np.linalg.solve(A_u, b_u)

            pred   = u @ V.T
            resid2 = (data - pred) ** 2
            train_chi2 = (
                (resid2[train_mask] / sigma[train_mask] ** 2).sum() / max(n_train, 1)
            )
            if abs(prev_chi2 - train_chi2) / max(prev_chi2, 1e-12) < tol:
                break
            prev_chi2 = train_chi2

        test_chi2 = (
            (resid2[test_mask] / sigma[test_mask] ** 2).sum() / max(n_test, 1)
        )
        if train_chi2 < best_train_chi2:
            best_train_chi2, best_test_chi2 = train_chi2, test_chi2
            best_u, best_V = u.copy(), V.copy()

    return best_u, best_V, best_train_chi2, best_test_chi2


# -- CV fold & rank sweep ----------------------------------------------
def _fit_one_gpt(data_mat, sigma_mat, fold_ids, K, f, reg):
    train_mask = (fold_ids != f)
    for _ in range(4):
        try:
            _, _, tr, te = gpt_als_fit(
                data_mat, sigma_mat, K=K, train_mask=train_mask, reg=reg,
                rng=np.random.default_rng(RANDOM_SEED + K * 100 + f),
            )
            return K, f, tr, te
        except LinAlgError:
            reg *= 10
    return K, f, 1e6, 1e6


def run_gpt_rank_sweep(data_mat, N_eff, label='', n_jobs=-1):
    """10-fold CV GPT rank sweep (K = 1-20)."""
    data_mat      = _resample_cols(data_mat, N_PX_SWEEP)
    n_rows, n_pix = data_mat.shape
    sigma_mat     = _poisson_sigma(data_mat, N_eff)
    sweep_reg     = max(ALS_REG, N_eff * 1e-6)

    rng_cv   = np.random.default_rng(RANDOM_SEED)
    flat_idx = rng_cv.permutation(n_rows * n_pix)
    fold_ids = (flat_idx % _N_FOLDS).reshape(n_rows, n_pix)

    ks = list(_K_RANGE_GPT)
    print(f'\n-- GPT {label}  ({n_rows}x{n_pix},  N_eff={N_eff:.1f},  reg={sweep_reg:.2e}) --')
    print(f'   Running {len(ks) * _N_FOLDS} jobs...')
    t0   = time.time()
    flat = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_gpt)(data_mat, sigma_mat, fold_ids, K, f, sweep_reg)
        for K in ks for f in range(_N_FOLDS)
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


# -- Plot --------------------------------------------------------------
def plot_gpt_sweep(cv_dict, mats_N, suptitle):
    ks    = list(_K_RANGE_GPT)
    x_pos = np.arange(len(ks))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(33, 10))
    fig.subplots_adjust(wspace=0.35)

    for ax, n_open in zip(axes.flat, [1, 2, 'all']):
        N_eff = mats_N[n_open]
        title_lbl = 'All configs' if n_open == 'all' else f'{n_open}-slit'

        tr_means = [np.mean(cv_dict[n_open][K]['train']) for K in ks]
        tr_stds  = [np.std (cv_dict[n_open][K]['train']) for K in ks]
        te_means = [np.mean(cv_dict[n_open][K]['test'])  for K in ks]
        te_stds  = [np.std (cv_dict[n_open][K]['test'])  for K in ks]

        ax.bar(x_pos - width/2, tr_means, width, yerr=tr_stds, capsize=3,
               color='steelblue', alpha=0.85, ecolor='navy',    label='Train')
        ax.bar(x_pos + width/2, te_means, width, yerr=te_stds, capsize=3,
               color='tomato',    alpha=0.85, ecolor='darkred', label='Test')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in ks], fontsize=9, rotation=45)
        ax.set_xlabel('GPT rank K', fontsize=12)
        ax.set_ylabel('chi2/pt', fontsize=12)
        ax.set_title(f'{title_lbl}  |  N_eff = {N_eff:.0f}', fontsize=13)
        ax.legend(fontsize=10, loc='lower left')

        # -- Inset --
        panel_inset_ks = _INSET_KS_1SLIT if n_open == 1 else _INSET_KS
        panel_table_ks = _TABLE_KS_1SLIT if n_open == 1 else _TABLE_KS
        inset_idx  = [i for i, k in enumerate(ks) if k in panel_inset_ks]
        inset_ks   = [ks[i] for i in inset_idx]
        inset_xpos = np.arange(len(inset_ks))

        axins = inset_axes(ax, width='55%', height='58%',
                           loc='upper right', borderpad=1.0)
        axins.bar(inset_xpos - width/2, [tr_means[i] for i in inset_idx], width,
                  yerr=[tr_stds[i] for i in inset_idx], capsize=3,
                  color='steelblue', alpha=0.85, ecolor='navy')
        axins.bar(inset_xpos + width/2, [te_means[i] for i in inset_idx], width,
                  yerr=[te_stds[i] for i in inset_idx], capsize=3,
                  color='tomato', alpha=0.85, ecolor='darkred')
        axins.axhline(1.0, color='gray', linestyle='--', linewidth=0.9)
        axins.set_xticks(inset_xpos)
        axins.set_xticklabels([str(k) for k in inset_ks], fontsize=8)
        axins.tick_params(axis='y', labelsize=8)
        axins.set_xlabel('K', fontsize=9)
        axins.set_ylabel('chi2/pt', fontsize=9)
        axins.set_title(f'K = {panel_inset_ks[0]}-{panel_inset_ks[-1]}', fontsize=9, pad=3)
        inset_vals = (
            [tr_means[i] - tr_stds[i] for i in inset_idx] +
            [te_means[i] - te_stds[i] for i in inset_idx] +
            [tr_means[i] + tr_stds[i] for i in inset_idx] +
            [te_means[i] + te_stds[i] for i in inset_idx]
        )
        lo, hi = min(inset_vals), max(inset_vals)
        pad_v  = max((hi - lo) * 0.15, 0.05)
        axins.set_ylim(lo - pad_v, hi + pad_v)

        _add_rank_table(ax, cv_dict[n_open], ks, table_ks=panel_table_ks)

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.show()


# -- Entry point -------------------------------------------------------
def _crop_bright_theory(mat, threshold=THEORY_CROP_THRESHOLD):
    if threshold <= 0.0:
        return mat
    bright = mat.mean(axis=0) >= threshold
    print(f'  Crop: keeping {bright.sum()}/{mat.shape[1]} columns '
          f'(threshold={threshold})')
    return mat[:, bright]


if __name__ == '__main__':
    theory_mats, theory_lbls, th_N_eff = build_theory_data(add_noise=True)

    # -- Plot theory matrices before running sweep --
    from data import plot_data
    plot_data(theory_mats, theory_lbls,
              f'Theory data (Poisson noise, N_eff={th_N_eff})')

    # -- Apply column crop --
    if THEORY_CROP_THRESHOLD > 0.0:
        all_mat_full = np.vstack([theory_mats[n] for n in [1, 2, 3, 4]])
        bright_mask = all_mat_full.mean(axis=0) >= THEORY_CROP_THRESHOLD
        print(f'  Common crop mask: {bright_mask.sum()}/{all_mat_full.shape[1]} columns kept')
        theory_mats = {n: theory_mats[n][:, bright_mask] for n in theory_mats}

    gpt_cv = {}
    for n_open in [1, 2]:
        gpt_cv[n_open] = run_gpt_rank_sweep(
            theory_mats[n_open], N_eff=th_N_eff,
            label=f'Theory (noisy)  {n_open}-slit',
        )
    all_mat = np.vstack([theory_mats[n] for n in [1, 2, 3, 4]])
    gpt_cv['all'] = run_gpt_rank_sweep(
        all_mat, N_eff=th_N_eff,
        label='Theory (noisy)  all-configs',
    )
    theory_mats_N = {1: th_N_eff, 2: th_N_eff, 'all': th_N_eff}

    plot_gpt_sweep(
        gpt_cv, theory_mats_N,
        suptitle=(
            f'GPT rank sweep -- s*phi_c*X*e model  |  '
            f'Theory + Poisson noise  |  {_N_FOLDS}-fold CV  |  N_eff={th_N_eff}\n'
            f'phases: {{0, pi/2, pi}}^4 = 81 patterns  |  '
            f'u_i[0]=1  |  Dashed: chi2/pt=1'
        ),
    )
