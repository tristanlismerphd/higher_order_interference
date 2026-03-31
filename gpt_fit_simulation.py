# ============================================================
#  GPT rank sweep — SIMULATION  (K=1-20, 10-fold CV, Poisson errors)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from gpt_fit_theory import run_rank_sweep, plot_sweep, _N_FOLDS, N_PX_SWEEP, _INSET_K_START
from data import build_simulation_data


if __name__ == '__main__':
    mats, lbls, mats_N = build_simulation_data()

    ex_cv = {}
    for n_open in [1, 2, 3, 4]:
        ex_cv[n_open] = run_rank_sweep(
            mats[n_open], N_eff=mats_N[n_open],
            label=f'Simulation  {n_open}-slit'
        )

    plot_sweep(
        ex_cv, mats, mats_N,
        suptitle=(f'Simulation GPT rank sweep  |  {_N_FOLDS}-fold CV  |  '
                  f'Poisson errors  |  phases: {{0, \u03c0/2, \u03c0}}\n'
                  f'Pixel axis resampled to {N_PX_SWEEP}  |  Dashed: \u03c7\u00b2/pt = 1  |  '
                  f'Inset: K \u2265 {_INSET_K_START}'),
        panel_prefix='Simulation',
    )
