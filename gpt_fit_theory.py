# ============================================================
#  GPT rank sweep — THEORETICAL  (K=1-25, 10-fold CV, Poisson errors)
#  Parallelised over (K, fold) combinations using joblib.
#  Prepends a pinned column of ones (unit effect) per Mazurek/Grabowecky.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from foundations import als_fit, RANDOM_SEED, ALS_REG
from data import build_simulation_data, build_theory_data

# ── Sweep parameters ──────────────────────────────────────────────────────────
_K_RANGE       = range(1, 26)
_N_FOLDS       = 10
_N_REST        = 4
P_FLOOR        = 0.01
N_PX_SWEEP     = 150
UNIT_SIGMA     = 1e-10

_INSET_K_START = {1: 4, 2: 4, 3: 9, 4: 16}

def _poisson_sigma(prob_mat, N_eff):
    return np.sqrt(np.maximum(prob_mat, P_FLOOR)) / np.sqrt(N_eff)

def _resample_cols(mat, n_out):
    n_in = mat.shape[1]
    if n_in == n_out:
        return mat
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    return np.array([np.interp(x_out, x_in, row) for row in mat])

def _fit_one(data_aug, sigma_aug, report_mask, fold_ids, K, f, sweep_reg, n_s):
    n_pix      = fold_ids.shape[1]
    data_mask  = (fold_ids != f)
    ones_mask  = np.ones((n_s, 1), dtype=bool)
    train_mask = np.hstack([ones_mask, data_mask])
    _, _, tr, te = als_fit(
        data_aug, sigma_aug, train_mask, K=K,
        n_restarts=_N_REST, reg=sweep_reg,
        rng=np.random.default_rng(RANDOM_SEED + K * 100 + f),
        report_mask=report_mask,
    )
    return K, f, tr, te

def run_rank_sweep(data_mat, N_eff, label='', n_jobs=-1):
    """10-fold CV rank sweep with pinned unit column, parallelised."""
    data_mat   = _resample_cols(data_mat, N_PX_SWEEP)
    n_s, n_pix = data_mat.shape

    ones_col  = np.ones((n_s, 1))
    data_aug  = np.hstack([ones_col, data_mat])

    sigma_aug       = _poisson_sigma(data_aug, N_eff)
    sigma_aug[:, 0] = UNIT_SIGMA

    report_mask = np.hstack([
        np.zeros((n_s, 1), dtype=bool),
        np.ones((n_s, n_pix), dtype=bool),
    ])

    sweep_reg = max(ALS_REG, N_eff * 1e-6)
    n_total   = n_s * n_pix
    rng_cv    = np.random.default_rng(RANDOM_SEED)
    perm      = rng_cv.permutation(n_total)
    fold_ids  = np.empty(n_total, dtype=int)
    for f in range(_N_FOLDS):
        fold_ids[perm[f::_N_FOLDS]] = f
    fold_ids = fold_ids.reshape(n_s, n_pix)

    print(f'\n── {label}  ({n_s}×{n_pix}+1,  N_eff={N_eff:.1f},  reg={sweep_reg:.2e}) ──')
    print(f'   [unit column prepended and pinned]')
    print(f'   Running {len(list(_K_RANGE)) * _N_FOLDS} jobs in parallel (n_jobs={n_jobs})...')

    t0   = time.time()
    jobs = [(K, f) for K in _K_RANGE for f in range(_N_FOLDS)]
    flat = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(data_aug, sigma_aug, report_mask, fold_ids, K, f, sweep_reg, n_s)
        for K, f in jobs
    )
    print(f'   Done in {time.time() - t0:.1f}s')

    results = {K: {'train': [], 'test': []} for K in _K_RANGE}
    for K, f, tr, te in flat:
        results[K]['train'].append(tr)
        results[K]['test'].append(te)

    for K in _K_RANGE:
        print(f'  K={K:>2}  train={np.mean(results[K]["train"]):.4f}  '
              f'test={np.mean(results[K]["test"]):.4f}')
    return results

def plot_sweep(cv_dict, mat_dict, mats_N, suptitle, panel_prefix):
    ks    = list(_K_RANGE)
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
        ax_img.set_title(f'{panel_prefix}  |  {n_open}-slit\n'
                         f'{mat.shape[0]} settings × {mat.shape[1]} px  '
                         f'(expected rank={n_open**2})',
                         fontsize=10)
        ax_img.set_xlabel('pixel index', fontsize=9)
        ax_img.set_ylabel('setting index', fontsize=9)
        ax_img.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.03, label='intensity')

        ax = axes[1, col]
        tr_means = [np.mean(cv_dict[n_open][K]['train']) for K in ks]
        tr_stds  = [np.std(cv_dict[n_open][K]['train'])  for K in ks]
        te_means = [np.mean(cv_dict[n_open][K]['test'])  for K in ks]
        te_stds  = [np.std(cv_dict[n_open][K]['test'])   for K in ks]

        ax.bar(x_pos - width/2, tr_means, width, yerr=tr_stds, capsize=3,
               color='steelblue', alpha=0.85, ecolor='navy', label='Train')
        ax.bar(x_pos + width/2, te_means, width, yerr=te_stds, capsize=3,
               color='tomato',    alpha=0.85, ecolor='darkred', label='Test')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
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
        pad = max((hi - lo) * 0.15, 0.05)
        axins.set_ylim(lo - pad, hi + pad)

    plt.suptitle(suptitle, fontsize=13)
    plt.show()


if __name__ == '__main__':
    _, _, mats_N             = build_simulation_data()
    theory_mats, _, th_N_eff = build_theory_data(add_noise=True)
    theory_mats_N = {n: th_N_eff for n in [1, 2, 3, 4]}

    th_cv = {}
    for n_open in [1, 2, 3, 4]:
        th_cv[n_open] = run_rank_sweep(
            theory_mats[n_open], N_eff=th_N_eff,
            label=f'Theory  {n_open}-slit'
        )

    plot_sweep(
        th_cv, theory_mats, theory_mats_N,
        suptitle=(f'Theoretical GPT rank sweep  |  {_N_FOLDS}-fold CV  |  '
                  f'Poisson noise  N_eff={th_N_eff}  |  phases: {{0, π/2, π}}\n'
                  f'{N_PX_SWEEP} pixels  |  Unit column pinned  |  '
                  f'Dashed: χ²/pt=1  |  Green: expected rank'),
        panel_prefix='Theory',
    )
