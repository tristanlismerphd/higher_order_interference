# higher_order_interference

Using GPTs to verify the principle of higher order interference (or lack thereof).

## What this project does

Tests whether quantum mechanics predicts the correct "GPT rank" of multi-slit interference data from a 4-rail optical interferometer.

**Key result to show:** `rank(D_all) = rank(D_2) = n²-1 = 15` for a 4-slit system.
- `D_all` = probability matrix stacking ALL slit configurations (1-, 2-, 3-, 4-slit rows)
- `D_2` = 2-slit configurations only
- `n²-1 = 15` for n=4 slits — the QM prediction

If rank < 15 fits just as well → classical or lower-order interference is sufficient.  
If rank 15 is needed → full quantum mechanics is required.

This is a theory-agnostic, GPT-rank-based test that is stronger than Sorkin's I₃=0 criterion, directly measuring the dimension of the state space.

**Experimental setup:** 4-rail interferometer, shutters 1–4, phase controllers Φ₁–₄ (phases 0, π/4, π/2), beam splitters H1–H8, emICCD camera (1024×1024).  
**Dataset:** `2026_04_09_3phase_0-pi4-pi2` — 16 shutter configs × 81 phase combos = 1296 files, named `{slit_idx}_{phase_idx}.npy`.

---

## Files

### Run these directly

| File | What it does |
|------|-------------|
| `rank_sweep_theory.py` | Generates simulated 4-slit Gaussian beam data with Poisson noise, runs the joint GPT rank sweep, and plots chi²/pt vs K. Use this to validate the fitter — expected minimum at K=15. Default N_eff=50000. |
| `rank_sweep_exp.py` | Loads real `.npy` files from the 2026-04-09 dataset, runs the joint rank sweep, and plots the same 3-panel chi²/pt figure. `EXP_N_EFF_SCALE=0.001` inflates sigma to suppress systematics. |
| `reading_data.py` | Standalone loader/visualiser — loads the 2026-04-09 dataset and plots the 4 probability matrices. No fitting. Good for a quick sanity check that data loaded correctly. |

### Library files (imported by the scripts above)

| File | What it does |
|------|-------------|
| `rank_sweep_joint.py` | Main fitter. Implements joint bilinear ALS: `D ≈ U @ V^T` where all n_open groups (1,2,3,4) share one V matrix (pixel effects). SVD initialisation, L2 regularisation, 10-fold CV. Entry point: `run_gpt_rank_sweep_joint(mats_dict, N_eff_dict, label, n_jobs)`. |
| `rank_sweep_gpt.py` | Original per-group fitter (with `u_i[0]=1` constraint). No longer the main fitter, but still imported for shared constants and plot helpers: `_K_RANGE_GPT`, `_N_FOLDS`, `N_PX_SWEEP`, `_add_rank_table`, `_poisson_sigma`, `_resample_cols`, etc. |
| `data.py` | Theory data generator. `build_theory_data(add_noise, N_eff)` returns `{n_open: prob_mat}` with Poisson noise. Uses 81 phase patterns ({0, π/4, π/2}⁴) and a Gaussian beam model with 4 slits. |
| `foundations.py` | Constants and low-level helpers: `RANDOM_SEED=42`, `ALS_REG=1e-6`, `_row_minmax()`, `als_fit()`. Note: contains old phase patterns ({0,π/2}⁴=16) and old beam params — only relevant if calling `als_fit` directly. |

### Cluster (Compute Canada, not currently in use)

| File | What it does |
|------|-------------|
| `cluster/run_gpt.py` | SLSQP fit for a single (group_id, K, fold) job — designed to run as a SLURM array task. |
| `cluster/gather.py` | Collects `.npz` results from cluster jobs and plots. |
| `cluster/submit.sh` | SLURM job array submission script. |

### Other files

| File | What it does |
|------|-------------|
| `proposal_rank_evidence.tex` | LaTeX supervisor proposal arguing for the GPT rank test. Includes theory derivation and results table (Table 2 needs experimental chi²/pt values once clean results are obtained). |
| `results_theory.png` | Chi²/pt vs K plot from theory sweep (K=15 minimum visible). |
| `results_experiment.png` | Chi²/pt vs K plot from experimental data. |
| `setup_diagram.png` | Diagram of the 4-rail interferometer setup. |

---

## Key physics

| Configuration | Expected rank |
|--------------|--------------|
| 1-slit only | 1 |
| Classical (no interference) | 4 |
| 2-path interference only | ~10 |
| Full QM (mixed-state input) | 15 = n²-1 |

---

## Last 5 changes

| Date | Change |
|------|--------|
| 2026-05-05 | Added detailed file descriptions and change log to README |
