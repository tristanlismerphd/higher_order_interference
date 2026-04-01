# ============================================================
#  rank_sweep_noisy.py
#  GPT rank sweep — theory data WITH Poisson shot noise.
#  Mimics a real experiment: quantum predictions + counting noise.
#  Run this to see the overfitting upturn at the true GPT rank.
#  (K=1-25, 10-fold CV, parallelised, unit column pinned)
# ============================================================
from gpt_fit_theory import (
    run_rank_sweep, plot_sweep, _N_FOLDS, N_PX_SWEEP, _INSET_K_START
)
from data import build_simulation_data, build_theory_data

if __name__ == '__main__':
    _, _, mats_N             = build_simulation_data()
    theory_mats, _, th_N_eff = build_theory_data(add_noise=True)
    theory_mats_N = {n: th_N_eff for n in [1, 2, 3, 4]}

    th_cv = {}
    for n_open in [1, 2, 3, 4]:
        th_cv[n_open] = run_rank_sweep(
            theory_mats[n_open], N_eff=th_N_eff,
            label=f'Theory (noisy)  {n_open}-slit'
        )

    plot_sweep(
        th_cv, theory_mats, theory_mats_N,
        suptitle=(f'GPT rank sweep — Theory + Poisson noise  |  {_N_FOLDS}-fold CV  |  '
                  f'N_eff={th_N_eff}  |  phases: {{0, π/2}}^4\n'
                  f'{N_PX_SWEEP} pixels  |  Unit column pinned  |  '
                  f'Dashed: χ²/pt=1  |  Green: expected rank'),
        panel_prefix='Theory (noisy)',
    )
