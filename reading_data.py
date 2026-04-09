# reading_data.py
# Load and plot intensity matrices from 2026_04_09_3phase_0-pi4-pi2 dataset.
# Files: {s_index}_{p_index}.npy  (1024x1024 camera frames, numpy binary)
# Shutter encoding: O=open, X=closed  (16 configs, index 0-15)
# Phase encoding: {0, pi/4, pi/2}^4 = 81 patterns (index 0-80)

import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

DATA_DIR = (
    '/Users/tristan_lismer/Desktop/PhD/Research/'
    'Higher-order interference/Data/exp_data/2026_04_09_3phase_0-pi4-pi2'
)

_N_PHASES = 81

_SLIT_CONFIGS = [
    (0,  'XXXX', 0), (1,  'XXOX', 1), (2,  'XXXO', 1),
    (3,  'XXOO', 2), (4,  'OXXX', 1), (5,  'OXOX', 2),
    (6,  'OXXO', 2), (7,  'OXOO', 3), (8,  'XOXX', 1),
    (9,  'XOOX', 2), (10, 'XOXO', 2), (11, 'XOOO', 3),
    (12, 'OOXX', 2), (13, 'OOOX', 3), (14, 'OOXO', 3),
    (15, 'OOOO', 4),
]

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
    total = float(col_sum.sum())
    return col_sum, total


def load_matrices(data_dir=DATA_DIR, n_jobs=-1):
    print('Loading data...')
    mats, N_effs = {}, {}
    for n_open, slit_idxs in sorted(_N_OPEN_TO_SLIT_IDXS.items()):
        tasks = [(s, p) for s in slit_idxs for p in range(_N_PHASES)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_load_one_row)(data_dir, s, p) for s, p in tasks
        )
        valid    = [(c, t) for c, t in results if c is not None]
        col_sums = np.array([c for c, t in valid])
        totals   = np.array([t for c, t in valid])
        prob_mat = col_sums / totals[:, None]
        mats[n_open]  = prob_mat
        N_effs[n_open] = float(np.median(totals))
        print(f'  n_open={n_open}: {prob_mat.shape[0]} rows, '
              f'median N_eff = {N_effs[n_open]:.3e}')
    print('Done.')
    return mats, N_effs


def _row_norm(mat):
    m = mat.astype(float).copy()
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (m - lo) / rng


def plot_matrices(mats, N_effs):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        'Experimental data matrices  |  row-normalised intensity\n'
        'phases: {0, \u03c0/4, \u03c0/2}\u2074 = 81 patterns  |  '
        '2026-04-09 dataset',
        fontsize=13,
    )
    for ax, n_open in zip(axes.flat, [1, 2, 3, 4]):
        mat = mats[n_open]
        im  = ax.imshow(_row_norm(mat), aspect='auto', origin='lower',
                        cmap='magma', vmin=0, vmax=1)
        ax.set_title(
            f'{n_open}-slit  |  {mat.shape[0]} settings x {mat.shape[1]} px\n'
            f'median N_eff = {N_effs[n_open]:.2e}',
            fontsize=11,
        )
        ax.set_xlabel('pixel index', fontsize=9)
        ax.set_ylabel('setting index', fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mats, N_effs = load_matrices(DATA_DIR)
    plot_matrices(mats, N_effs)
