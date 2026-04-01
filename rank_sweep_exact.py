# ============================================================
#  rank_sweep_exact.py
#  GPT rank sweep — EXACT noiseless simulation data.
#  No shot noise: perfect quantum predictions.
#  (K=1-25, 10-fold CV, parallelised, unit column pinned)
# ============================================================
from gpt_fit_theory import (
    run_rank_sweep, plot_sweep, _N_FOLDS, N_PX_SWEEP, _INSET_K_START
)
from data import build_simulation_data

if __name__ == '__main__':
    mats, lbls, mats_N = build_simulation_data()

    ex_cv = {}
    for n_open in [1, 2, 3, 4]:
        ex_cv[n_open] = run_rank_sweep(
            mats[n_open], N_eff=mats_N[n_open],
            label=f'Exact simulation  {n_open}-slit'
        )

    plot_sweep(
        ex_cv, mats, mats_N,
        suptitle=(f'GPT rank sweep — Exact noiseless simulation  |  {_N_FOLDS}-fold CV  |  '
                  f'phases: {{0, π/2}}^4\n'
                  f'{N_PX_SWEEP} pixels  |  Unit column pinned  |  '
                  f'Dashed: χ²/pt=1  |  Green: expected rank'),
        panel_prefix='Exact',
    )
