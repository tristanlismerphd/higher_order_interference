# rank_sweep_theory.py
# Joint GPT rank sweep on THEORETICAL data.
# Phases: {0, pi/4, pi/2}^4 = 81 patterns  (matches 2026-04-09 experiment)
# Uses the same joint fitter as rank_sweep_exp.py:
#   all n_open groups share one V (pixel effects) and are fitted simultaneously.
import numpy as np
import matplotlib.pyplot as plt
from foundations import _row_minmax
from data import build_theory_data
from rank_sweep_gpt import (
    _INSET_KS, _INSET_KS_1SLIT,
    _K_RANGE_GPT, _N_FOLDS,
    _TABLE_KS, _TABLE_KS_1SLIT, _add_rank_table,
)
from rank_sweep_joint import run_gpt_rank_sweep_joint
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

THEORY_N_EFF = 50000   # Poisson noise level for simulated data


def plot_theory_matrices(theory_mats, N_eff):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        'Theory data matrices  |  row-normalised intensity\n'
        'phases: {0, \u03c0/4, \u03c0/2}\u2074 = 81 patterns  |  Poisson noise',
        fontsize=13,
    )
    for ax, n_open in zip(axes.flat, [1, 2, 3, 4]):
        mat = _row_minmax(theory_mats[n_open])
        im  = ax.imshow(mat, aspect='auto', origin='lower',
                        cmap='magma', vmin=0, vmax=1)
        ax.set_title(
            f'{n_open}-slit  |  {mat.shape[0]} settings x {mat.shape[1]} px\n'
            f'N_eff = {N_eff:.2e}',
            fontsize=11,
        )
        ax.set_xlabel('pixel index', fontsize=9)
        ax.set_ylabel('setting index', fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    plt.tight_layout()
    plt.show()


def plot_theory_sweep(cv_dict, N_eff, suptitle):
    ks    = list(_K_RANGE_GPT)
    x_pos = np.arange(len(ks))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(33, 10))
    fig.subplots_adjust(wspace=0.35)
    for ax, n_open in zip(axes.flat, [1, 2, 'all']):
        title_lbl = 'All configs' if n_open == 'all' else f'{n_open}-slit'
        tr_means  = [np.mean(cv_dict[n_open][K]['train']) for K in ks]
        tr_stds   = [np.std (cv_dict[n_open][K]['train']) for K in ks]
        te_means  = [np.mean(cv_dict[n_open][K]['test'])  for K in ks]
        te_stds   = [np.std (cv_dict[n_open][K]['test'])  for K in ks]
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

        panel_inset_ks = _INSET_KS_1SLIT if n_open == 1 else _INSET_KS
        panel_table_ks = _TABLE_KS_1SLIT  if n_open == 1 else _TABLE_KS
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
        axins.set_title(f'K = {panel_inset_ks[0]}-{panel_inset_ks[-1]}',
                        fontsize=9, pad=3)
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


if __name__ == '__main__':
    theory_mats, _, N_eff = build_theory_data(add_noise=True, N_eff=THEORY_N_EFF)

    plot_theory_matrices(theory_mats, N_eff)

    N_eff_dict = {n: N_eff for n in [1, 2, 3, 4]}
    gpt_cv = run_gpt_rank_sweep_joint(
        theory_mats, N_eff_dict,
        label='Theory (joint)',
    )

    plot_theory_sweep(
        gpt_cv, N_eff,
        suptitle=(
            f'GPT rank sweep -- Theory + Poisson noise  |  {_N_FOLDS}-fold CV  |  joint fit\n'
            f'phases: {{0, pi/4, pi/2}}^4 = 81 patterns  |  '
            f'shared V across all configs  |  N_eff={N_eff}  |  Dashed: chi2/pt=1'
        ),
    )
