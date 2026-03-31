# ============================================================
#  Data — build simulation & theory intensity matrices, print summary
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as _prod
from foundations import _row_minmax

# ── Beam / grid parameters ────────────────────────────────────────────────────
_SLIT_X      = np.array([-0.05, -0.015, 0.015, 0.05])
_BEAM_RADIUS = 0.5
_KX_LIST     = [-20, -10, 10, 20]
_NUM_PIXELS  = 150
_x_grid      = np.linspace(-1, 1, _NUM_PIXELS)

# ── Shutter labels (O=open, X=closed) ────────────────────────────────────────
shutter_labels = [
    'X,X,X,X', 'X,X,O,X', 'X,X,X,O', 'X,X,O,O',
    'O,X,X,X', 'O,X,O,X', 'O,X,X,O', 'O,X,O,O',
    'X,O,X,X', 'X,O,O,X', 'X,O,X,O', 'X,O,O,O',
    'O,O,X,X', 'O,O,O,X', 'O,O,X,O', 'O,O,O,O'
]

# ── Phase grid: {0, π/2, π} varied only for open slits ───────────────────────
_PHASE_SET  = [0.0, np.pi / 2, np.pi]
_PHASE_STRS = ['0', r'$\pi/2$', r'$\pi$']

def _build_phase_label(open_slits, phase_combo):
    strs = ['0'] * 4
    for idx, k in enumerate(open_slits):
        strs[k] = _PHASE_STRS[_PHASE_SET.index(phase_combo[idx])]
    return ','.join(strs)

def _simulate_row(open_slits, phase_combo):
    field = sum(
        np.exp(-2 * (_x_grid - _SLIT_X[k])**2 / _BEAM_RADIUS**2)
        * np.exp(1j * (phase_combo[idx] + _KX_LIST[k] * _x_grid))
        for idx, k in enumerate(open_slits)
    )
    return np.abs(field)**2

def build_simulation_data():
    """Build simulated (experimental-mimicking) intensity matrices."""
    mats, lbls, mats_N = {}, {}, {}
    for n_open in [1, 2, 3, 4]:
        rows, row_labels = [], []
        for sl in shutter_labels:
            parts      = sl.split(',')
            open_slits = [i for i, c in enumerate(parts) if c == 'O']
            if len(open_slits) != n_open:
                continue
            for phase_combo in _prod(_PHASE_SET, repeat=n_open):
                rows.append(_simulate_row(open_slits, phase_combo))
                row_labels.append(f'{sl} | {_build_phase_label(open_slits, phase_combo)}')
        mat  = np.array(rows)
        mins = mat.min(axis=1, keepdims=True)
        rng  = mat.max(axis=1, keepdims=True) - mins
        rng[rng == 0] = 1.0
        mats[n_open]   = (mat - mins) / rng
        lbls[n_open]   = row_labels
        mats_N[n_open] = 1000.0
    return mats, lbls, mats_N

def build_theory_data():
    """Build theoretical intensity matrices."""
    theory_mats, theory_lbls = {}, {}
    for n_open in [1, 2, 3, 4]:
        rows, row_labels = [], []
        for sl in shutter_labels:
            parts      = sl.split(',')
            open_slits = [i for i, c in enumerate(parts) if c == 'O']
            if len(open_slits) != n_open:
                continue
            for phase_combo in _prod(_PHASE_SET, repeat=n_open):
                field = sum(
                    np.exp(-2 * (_x_grid - _SLIT_X[k])**2 / _BEAM_RADIUS**2)
                    * np.exp(1j * (phase_combo[idx] + _KX_LIST[k] * _x_grid))
                    for idx, k in enumerate(open_slits)
                )
                rows.append(np.abs(field)**2)
                row_labels.append(f'{sl} | {_build_phase_label(open_slits, phase_combo)}')
        theory_mats[n_open] = _row_minmax(np.array(rows))
        theory_lbls[n_open] = row_labels
    return theory_mats, theory_lbls

def print_summary(mats, theory_mats, mats_N):
    print('\n=== Simulation data ===')
    for n_open in [1, 2, 3, 4]:
        print(f'  {n_open}-slit: {mats[n_open].shape}  N_eff={mats_N[n_open]:.0f}')
    print('\n=== Theory data ===')
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
    plt.suptitle(f'{title_suffix}  |  phases: {{0, \u03c0/2, \u03c0}} (open slits only)',
                 fontsize=16, y=1.002)
    plt.show()


if __name__ == '__main__':
    mats, lbls, mats_N       = build_simulation_data()
    theory_mats, theory_lbls = build_theory_data()
    print_summary(mats, theory_mats, mats_N)
    plot_data(mats, lbls, 'Simulation data')
    plot_data(theory_mats, theory_lbls, 'Theoretical data')
