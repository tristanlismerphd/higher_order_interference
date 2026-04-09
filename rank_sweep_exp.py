# rank_sweep_exp.py
# GPT rank sweep on EXPERIMENTAL data.
# Last change: switched to 2026-04-09 dataset (.npy files, phases {0, pi/4, pi/2}^4)
# Files: {slit_idx}_{phase_idx}.npy  (1024x1024 camera frames)
# Slit encoding: O=open, X=closed
# Phase encoding: 81 patterns = {0, pi/4, pi/2}^4
# Plots the 4 data matrices first, then runs the sweep.
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rank_sweep_gpt import (
    _INSET_KS, _INSET_KS_1SLIT,
    _K_RANGE_GPT, _N_FOLDS, _poisson_sigma, _resample_cols, N_PX_SWEEP,
    _TABLE_KS, _TABLE_KS_1SLIT, _add_rank_table,
)
from rank_sweep_joint import run_gpt_rank_sweep_joint
from rank_sweep_structured import run_gpt_rank_sweep_structured, _K_RANGE_STRUCT, N2

# Scale factor applied to N_eff before fitting (tune to get chi2/pt ~ 1 at true rank)
EXP_N_EFF_SCALE = 0.001

# Crop threshold: columns where mean intensity across ALL rows < this value are dropped.
# exp_mats rows sum to 1 over 1024 px, so mean col ~ 0.001; set ~2-3x that to trim dark edges.
# Set to 0.0 to disable cropping.
EXP_CROP_THRESHOLD = 0.0

# SET THIS to your local data directory
DATA_DIR = (
    '/Users/tristan_lismer/Desktop/PhD/Research/'
    'Higher-order interference/Data/exp_data/2026_04_09_3phase_0-pi4-pi2'
)

# Slit-config index -> (pattern, n_open).  O=open, X=closed.
_SLIT_CONFIGS = [
    (0,  'XXXX', 0), (1,  'XXOX', 1), (2,  'XXXO', 1),
    (3,  'XXOO', 2), (4,  'OXXX', 1), (5,  'OXOX', 2),
    (6,  'OXXO', 2), (7,  'OXOO', 3), (8,  'XOXX', 1),
    (9,  'XOOX', 2), (10, 'XOXO', 2), (11, 'XOOO', 3),
    (12, 'OOXX', 2), (13, 'OOOX', 3), (14, 'OOXO', 3),
    (15, 'OOOO', 4),
]
_N_PHASES = 81
_N_OPEN_TO_SLIT_IDXS = {}
for _idx, _pat, _nop in _SLIT_CONFIGS:
    if _nop > 0:
        _N_OPEN_TO_SLIT_IDXS.setdefault(_nop, []).append(_idx)


def _load_one_row(data_dir, slit_idx, phase_idx):
    path = os.path.join(data_dir, f'{slit_idx}_{phase_idx}.npy')
    if not os.path.exists(path):
        return None, None
    frame = np.load(path).astype(float)
    col_sum = frame.sum(axis=0)
    return col_sum, float(col_sum.sum())


def load_exp_matrices(data_dir=DATA_DIR, n_jobs=-1):
    """Returns (exp_mats, exp_N_eff, row_configs_dict).
    row_configs_dict[n_open] = [(slit_idx, phase_idx), ...] for each valid row."""
    print('Loading experimental data...')
    t0 = time.time()
    exp_mats, exp_N_eff, row_configs_dict = {}, {}, {}
    for n_open, slit_idxs in sorted(_N_OPEN_TO_SLIT_IDXS.items()):
        tasks = [(s, p) for s in slit_idxs for p in range(_N_PHASES)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_load_one_row)(data_dir, s, p) for s, p in tasks
        )
        valid_idx = [i for i, (c, t) in enumerate(results) if c is not None]
        valid     = [results[i] for i in valid_idx]
        col_sums  = np.array([c for c, t in valid])
        totals    = np.array([t for c, t in valid])
        prob_mat  = col_sums / totals[:, None]
        exp_mats[n_open]        = prob_mat
        exp_N_eff[n_open]       = float(np.median(totals))
        row_configs_dict[n_open] = [tasks[i] for i in valid_idx]
        print(f'  n_open={n_open}: {prob_mat.shape[0]} rows, '
              f'median N_eff = {exp_N_eff[n_open]:.3e}')
    print(f'  Done in {time.time() - t0:.1f}s')
    return exp_mats, exp_N_eff, row_configs_dict


def _row_norm(mat):
    m = mat.astype(float).copy()
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (m - lo) / rng


def plot_exp_matrices(exp_mats, exp_N_eff):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        'Experimental data matrices  |  row-normalised intensity\n'
        'Each row = one (slit config, phase) setting',
        fontsize=13,
    )
    for ax, n_open in zip(axes.flat, [1, 2, 3, 4]):
        mat = exp_mats[n_open]
        im  = ax.imshow(_row_norm(mat), aspect='auto', origin='lower',
                        cmap='magma', vmin=0, vmax=1)
        ax.set_title(
            f'{n_open}-slit  |  {mat.shape[0]} settings x {mat.shape[1]} px\n'
            f'median N_eff = {exp_N_eff[n_open]:.2e}',
            fontsize=11,
        )
        ax.set_xlabel('pixel index', fontsize=9)
        ax.set_ylabel('setting index', fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    plt.tight_layout()
    plt.show()


def plot_exp_sweep(cv_dict, exp_N_eff, suptitle):
    ks    = list(_K_RANGE_GPT)
    x_pos = np.arange(len(ks))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(33, 10))
    fig.subplots_adjust(wspace=0.35)
    for ax, n_open in zip(axes.flat, [1, 2, 'all']):
        N_eff    = exp_N_eff[n_open]
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
        ax.set_title(f'{title_lbl}  |  N_eff = {N_eff:.2e}', fontsize=13)
        ax.legend(fontsize=10, loc='lower left')
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


def _crop_bright(mat, threshold=EXP_CROP_THRESHOLD):
    if threshold <= 0.0:
        return mat
    mean_col = mat.mean(axis=0)
    bright = mean_col >= threshold
    print(f'  Cropping: keeping {bright.sum()}/{len(bright)} columns '
          f'(threshold={threshold})')
    return mat[:, bright]


def plot_structured_sweep(cv_dict, N_eff_display, suptitle):
    """Plot structured-model sweep results (K=1..16)."""
    import matplotlib.pyplot as plt
    ks    = list(_K_RANGE_STRUCT)
    x_pos = np.arange(len(ks))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(33, 10))
    fig.subplots_adjust(wspace=0.35)
    for ax, n_open in zip(axes.flat, [1, 2, 'all']):
        N_eff     = N_eff_display[n_open]
        title_lbl = 'All configs' if n_open == 'all' else f'{n_open}-slit'
        tr_means  = [np.mean(cv_dict[n_open][K]['train']) for K in ks]
        tr_stds   = [np.std (cv_dict[n_open][K]['train']) for K in ks]
        te_means  = [np.mean(cv_dict[n_open][K]['test'])  for K in ks]
        te_stds   = [np.std (cv_dict[n_open][K]['test'])  for K in ks]
        ax.bar(x_pos - width/2, tr_means, width, yerr=tr_stds, capsize=3,
               color='steelblue', alpha=0.85, ecolor='navy', label='Train')
        ax.bar(x_pos + width/2, te_means, width, yerr=te_stds, capsize=3,
               color='tomato',    alpha=0.85, ecolor='darkred', label='Test')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x_pos[14] + 0.5, color='green', linestyle=':', linewidth=1.2,
                   label='K=15 (QM)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in ks], fontsize=9)
        ax.set_xlabel('GPT rank K  (HS basis modes)', fontsize=12)
        ax.set_ylabel('chi2/pt', fontsize=12)
        ax.set_title(f'{title_lbl}  |  N_eff = {N_eff:.2e}', fontsize=13)
        ax.legend(fontsize=10, loc='upper right')
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    exp_mats, exp_N_eff, row_configs_dict = load_exp_matrices(DATA_DIR)
    plot_exp_matrices(exp_mats, exp_N_eff)
    all_mat_full = np.vstack([exp_mats[n] for n in [1, 2, 3, 4]])
    col_means = all_mat_full.mean(axis=0)
    print(f'  Column mean intensity: min={col_means.min():.5f}, '
          f'max={col_means.max():.5f}, median={np.median(col_means):.5f}, '
          f'p90={np.percentile(col_means, 90):.5f}')
    if EXP_CROP_THRESHOLD > 0.0:
        all_mat_full = np.vstack([exp_mats[n] for n in [1, 2, 3, 4]])
        bright_mask = all_mat_full.mean(axis=0) >= EXP_CROP_THRESHOLD
        print(f'  Common crop mask: {bright_mask.sum()}/1024 columns kept')
        exp_mats = {n: exp_mats[n][:, bright_mask] for n in exp_mats}
        plot_exp_matrices(exp_mats, exp_N_eff)

    N_eff_scaled = {n: exp_N_eff[n] * EXP_N_EFF_SCALE for n in [1, 2, 3, 4]}
    all_N_eff    = float(np.mean(list(exp_N_eff.values())))

    # -- Structured sweep (shared rho + shared V, T_s from physics) -----------
    struct_cv = run_gpt_rank_sweep_structured(
        exp_mats, N_eff_scaled, row_configs_dict,
        label='Experimental',
    )
    plot_structured_sweep(
        struct_cv,
        {1: N_eff_scaled[1], 2: N_eff_scaled[2], 'all': all_N_eff * EXP_N_EFF_SCALE},
        suptitle=(
            f'Structured GPT sweep -- Experimental data  |  {_N_FOLDS}-fold CV\n'
            f'T_s from known slit mask + phases  |  shared rho + shared V  |  '
            f'K=4: classical  K=15: full QM  |  Dashed: chi2/pt=1'
        ),
    )
