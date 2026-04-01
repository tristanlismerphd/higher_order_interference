# ============================================================
#  Data — build simulation & theory intensity matrices
#  Per-slit Gaussians with spatial offsets produce the slant seen in
#  experimental data (each slit peaks at a different detector position).
#  Phases follow {0, π/2}^4 = 16 patterns for ALL slits.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as _prod
from foundations import _row_minmax, RANDOM_SEED

# ── Beam / grid parameters ───────────────────────────────────────────────────
BEAM_RADIUS = 1.0
SLIT_X      = np.array([-0.3, -0.1, 0.1, 0.3])
KX_LIST     = [-20, -10, 10, 20]
NUM_PIXELS  = 500
x_grid      = np.linspace(-3 * BEAM_RADIUS, 3 * BEAM_RADIUS, NUM_PIXELS)

# ── Phase patterns: {0, π/2}^4 = 16 settings for ALL slits ─────────────────
phase_patterns = list(_prod([0.0, np.pi / 2], repeat=4))

def _phase_label(combo):
    return ','.join('0' if p == 0 else 'π/2' for p in combo)

# ── Shutter labels (O=open, X=closed) ────────────────────────────────────────
shutter_labels = [
    'X,X,X,X', 'X,X,O,X', 'X,X,X,O', 'X,X,O,O',
    'O,X,X,X', 'O,X,O,X', 'O,X,X,O', 'O,X,O,O',
    'X,O,X,X', 'X,O,O,X', 'X,O,X,O', 'X,O,O,O',
    'O,O,X,X', 'O,O,O,X', 'O,O,X,O', 'O,O,O,O'
]

def _amplitudes(sl):
    return tuple(0 if b == 'X' else 1 for b in sl.split(','))

def _simulate_row(A_tuple, phase_combo):
    field = sum(
        A * np.exp(-2 * (x_grid - SLIT_X[k])**2 / BEAM_RADIUS**2)
        * np.exp(1j * (ph + KX_LIST[k] * x_grid))
        for k, (A, ph) in enumerate(zip(A_tuple, phase_combo))
    )
    return np.abs(field)**2

def build_simulation_data():
    mats, lbls, mats_N = {}, {}, {}
    for n_open in [1, 2, 3, 4]:
        rows, row_labels = [], []
        for sl in shutter_labels:
            A = _amplitudes(sl)
            if sum(A) != n_open:
                continue
            for phase_combo in phase_patterns:
                rows.append(_simulate_row(A, phase_combo))
                row_labels.append(f'{sl} | {_phase_label(phase_combo)}')
        mat  = np.array(rows)
        mins = mat.min(axis=1, keepdims=True)
        rng  = mat.max(axis=1, keepdims=True) - mins
        rng[rng == 0] = 1.0
        mats[n_open]   = (mat - mins) / rng
        lbls[n_open]   = row_labels
        mats_N[n_open] = 1000.0
    return mats, lbls, mats_N

def build_theory_data(add_noise=True, N_eff=50000):
    rng = np.random.default_rng(RANDOM_SEED)
    theory_mats, theory_lbls = {}, {}
    for n_open in [1, 2, 3, 4]:
        rows, row_labels = [], []
        for sl in shutter_labels:
            A = _amplitudes(sl)
            if sum(A) != n_open:
                continue
            for phase_combo in phase_patterns:
                rows.append(_simulate_row(A, phase_combo))
                row_labels.append(f'{sl} | {_phase_label(phase_combo)}')

        exact    = np.array(rows)
        row_sums = exact.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob     = exact / row_sums

        if add_noise:
            counts = rng.poisson(prob * N_eff)
            theory_mats[n_open] = np.maximum(counts / N_eff, 0.0)
        else:
            theory_mats[n_open] = _row_minmax(exact)

        theory_lbls[n_open] = row_labels
    return theory_mats, theory_lbls, N_eff

def print_summary(mats, theory_mats, mats_N, theory_N_eff):
    print('\n=== Simulation data ===')
    for n_open in [1, 2, 3, 4]:
        print(f'  {n_open}-slit: {mats[n_open].shape}  N_eff={mats_N[n_open]:.0f}')
    print(f'\n=== Theory data (N_eff={theory_N_eff}) ===')
    for n_open in [1, 2, 3, 4]:
        print(f'  {n_open}-slit: {theory_mats[n_open].shape}')

def plot_data(mats, lbls, title_suffix):
    fig, axes = plt.subplots(2, 2, figsize=(32, 80))
    fig.subplots_adjust(hspace=0.01, wspace=0.3)
    for ax, n_open in zip(axes.flat, [1, 2, 3, 4]):
        mat = mats[n_open]
        lbl = lbls[n_open]
        im  = ax.imshow(mat, aspect='auto', origin='lower',
                        cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'{n_open} slit(s) open  [{mat.shape[0]} settings × {mat.shape[1]} px]',
                     fontsize=16)
        ax.set_xlabel('pixel index', fontsize=13)
        ax.set_ylabel('setting', fontsize=13)
        ax.set_yticks(np.arange(len(lbl)))
        ax.set_yticklabels(lbl, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='row-norm. intensity')
    plt.suptitle(f'{title_suffix}  |  phases: {{0, π/2}}^4',
                 fontsize=16, y=1.002)
    plt.show()


if __name__ == '__main__':
    mats, lbls, mats_N                = build_simulation_data()
    theory_mats, theory_lbls, th_Neff = build_theory_data(add_noise=True)
    print_summary(mats, theory_mats, mats_N, th_Neff)
    plot_data(mats, lbls, 'Simulation data')
    plot_data(theory_mats, theory_lbls, f'Theoretical data (Poisson noise, N_eff={th_Neff})')
