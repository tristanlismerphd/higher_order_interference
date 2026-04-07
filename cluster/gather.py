"""
gather.py  —  aggregate results, produce:
  1. intensity_all.png         theoretical intensity matrices (2x3, 5 panels)
  2. rank_selection_all.png    chi2/pt vs rank with insets + tables (2x3, 5 panels)

Usage:
    python gather.py --outdir results/
"""
import argparse, os
import numpy as np
from itertools import product as _prod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="results")
args = parser.parse_args()

N_RANKS = 20
N_FOLDS = 8

# group_id -> slit count label
SLITS_OPEN_MAP = {0: 4, 1: 3, 2: 2, 3: 1, 4: 'all'}
DISPLAY_ORDER  = [3, 2, 1, 0, 4]   # 1-slit, 2-slit, 3-slit, 4-slit, all

# Inset and table K ranges — special-cased for 1-slit (group 3)
INSET_KS       = {3: list(range(1, 6))}
INSET_KS_DEF   = list(range(12, 19))
TABLE_KS       = {3: [1, 2, 3, 4, 5]}
TABLE_KS_DEF   = [14, 15, 16, 17, 18]

# ── Beam parameters ───────────────────────────────────────────────────────────
BEAM_RADIUS = 0.5
KX_LIST     = [-20, -10, 10, 20]
SLIT_X      = np.array([-0.05, -0.015, 0.015, 0.05])
NUM_PIXELS  = 150
PHASES_LIST = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
x_grid      = np.linspace(-1, 1, NUM_PIXELS)

SHUTTER_PATTERNS = [
    'XXXX', 'XXOX', 'XXXO', 'XXOO', 'OXXX', 'OXOX', 'OXXO', 'OXOO',
    'XOXX', 'XOOX', 'XOXO', 'XOOO', 'OOXX', 'OOOX', 'OOXO', 'OOOO'
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def row_minmax(mat):
    out  = mat.astype(float).copy()
    mins = out.min(axis=1, keepdims=True)
    rngs = out.max(axis=1, keepdims=True) - mins
    rngs[rngs == 0] = 1.0
    return (out - mins) / rngs


def simulate(slit_count):
    """Simulate intensity matrix for given slit count (or 'all')."""
    rows = []
    for sh in SHUTTER_PATTERNS:
        open_slits = tuple(i for i, ch in enumerate(sh) if ch == 'O')
        if slit_count == 'all':
            if len(open_slits) == 0:
                continue
        elif len(open_slits) != slit_count:
            continue
        for phases in _prod(PHASES_LIST, repeat=4):
            field = sum(
                np.exp(-2 * (x_grid - SLIT_X[k]) ** 2 / BEAM_RADIUS ** 2)
                * np.exp(1j * (phases[k] + KX_LIST[k] * x_grid))
                for k in open_slits
            )
            rows.append(np.abs(field) ** 2)
    return row_minmax(np.array(rows))


def add_rank_table(ax, res, ks, table_ks):
    cell_text = []
    for K in table_ks:
        if K not in res:
            continue
        r = res[K]
        cell_text.append([
            str(K),
            f"{r['mean_train']:.4f} \u00b1 {r['std_train']:.4f}",
            f"{r['mean_test']:.4f} \u00b1 {r['std_test']:.4f}",
        ])
    if not cell_text:
        return
    tbl = ax.table(
        cellText=cell_text,
        colLabels=['K', 'Train (mean\u00b1std)', 'Test (mean\u00b1std)'],
        loc='bottom',
        bbox=[0, -0.54, 1, 0.32],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)


# ── Load results ──────────────────────────────────────────────────────────────
all_results = {g: {} for g in range(5)}
missing = []

for g in range(5):
    for k in range(1, N_RANKS + 1):
        fold_train, fold_test = [], []
        best_x, best_te, best_fold = None, np.inf, -1
        n_s_saved = num_px_saved = None

        for fold in range(N_FOLDS):
            path = os.path.join(args.outdir, f"result_group{g}_k{k}_fold{fold}.npz")
            if not os.path.exists(path):
                missing.append(path)
                continue
            d = np.load(path, allow_pickle=True)
            if not bool(d["success"]):
                continue
            tr = float(d["train_err"])
            te = float(d["test_err"])
            fold_train.append(tr)
            fold_test.append(te)
            if n_s_saved is None:
                n_s_saved    = int(d["n_s"])
                num_px_saved = int(d["num_pixels"])
            if te < best_te:
                best_te, best_x, best_fold = te, d["x_opt"].copy(), fold

        if fold_train:
            all_results[g][k] = dict(
                mean_train=np.mean(fold_train), std_train=np.std(fold_train),
                mean_test =np.mean(fold_test),  std_test =np.std(fold_test),
                best_x=best_x, best_fold=best_fold, best_te=best_te,
                n_s=n_s_saved, num_pixels=num_px_saved,
            )

# ── Summary ───────────────────────────────────────────────────────────────────
for g in range(5):
    lbl = SLITS_OPEN_MAP[g]
    print(f"\n=== Group {g}  ({lbl} slit(s) open) ===")
    print(f"  {'Rank':>6}  {'Train chi2':>12}  {'Test chi2':>12}  {'Std':>10}")
    for k in range(1, N_RANKS + 1):
        if k not in all_results[g]:
            continue
        r = all_results[g][k]
        print(f"  {k:>6}  {r['mean_train']:>12.5f}  {r['mean_test']:>12.5f}  {r['std_test']:>10.5f}")

if missing:
    print(f"\nWARNING: {len(missing)} missing files (jobs still running?)")

# ── Plot 1: intensity matrices ────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(33, 18))
fig1.subplots_adjust(hspace=0.3, wspace=0.35)
axes1.flat[5].set_visible(False)

for ax, g in zip(axes1.flat, DISPLAY_ORDER):
    slit_lbl = SLITS_OPEN_MAP[g]
    title_lbl = 'All configs' if slit_lbl == 'all' else f'{slit_lbl}-slit'
    mat = simulate(slit_lbl)
    im  = ax.imshow(mat, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    ax.set_title(f'{title_lbl}  [{mat.shape[0]} settings x {mat.shape[1]} px]', fontsize=13)
    ax.set_xlabel('pixel index', fontsize=10)
    ax.set_ylabel('setting index', fontsize=10)
    ax.set_yticks([])
    fig1.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

fig1.suptitle(
    'Theoretical intensity matrices  |  phases: {0, pi/2, pi, 3pi/2}^4 = 256 patterns',
    fontsize=14, y=1.01,
)
intens_plot = os.path.join(args.outdir, "intensity_all.png")
fig1.savefig(intens_plot, dpi=150, bbox_inches='tight')
print(f"\nIntensity plot saved -> {intens_plot}")

# ── Plot 2: rank selection ────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(33, 18))
fig2.subplots_adjust(hspace=0.75, wspace=0.35)
axes2.flat[5].set_visible(False)

ks_all = list(range(1, N_RANKS + 1))
xpos   = np.arange(len(ks_all))
w      = 0.35

for ax, g in zip(axes2.flat, DISPLAY_ORDER):
    res      = all_results[g]
    slit_lbl = SLITS_OPEN_MAP[g]
    title_lbl = 'All configs' if slit_lbl == 'all' else f'{slit_lbl}-slit'

    if not res:
        ax.set_title(f'{title_lbl} — no data')
        continue

    ks_avail = [k for k in ks_all if k in res]
    xpos_av  = np.array([ks_all.index(k) for k in ks_avail])

    tr_means = [res[k]['mean_train'] for k in ks_avail]
    tr_stds  = [res[k]['std_train']  for k in ks_avail]
    te_means = [res[k]['mean_test']  for k in ks_avail]
    te_stds  = [res[k]['std_test']   for k in ks_avail]

    ax.bar(xpos_av - w/2, tr_means, w, yerr=tr_stds, capsize=3,
           color='steelblue', alpha=0.85, ecolor='navy',    label='Train')
    ax.bar(xpos_av + w/2, te_means, w, yerr=te_stds, capsize=3,
           color='tomato',    alpha=0.85, ecolor='darkred', label='Test')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(k) for k in ks_all], fontsize=8, rotation=45)
    ax.set_xlabel('GPT rank K', fontsize=11)
    ax.set_ylabel('chi2/pt', fontsize=11)
    ax.set_title(title_lbl, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')

    # Inset
    panel_inset_ks = INSET_KS.get(g, INSET_KS_DEF)
    panel_table_ks = TABLE_KS.get(g, TABLE_KS_DEF)
    inset_idx  = [i for i, k in enumerate(ks_avail) if k in panel_inset_ks]
    inset_ks   = [ks_avail[i] for i in inset_idx]
    inset_xpos = np.arange(len(inset_ks))

    if inset_idx:
        axins = inset_axes(ax, width='55%', height='58%', loc='upper right', borderpad=1.0)
        axins.bar(inset_xpos - w/2, [tr_means[i] for i in inset_idx], w,
                  yerr=[tr_stds[i] for i in inset_idx], capsize=3,
                  color='steelblue', alpha=0.85, ecolor='navy')
        axins.bar(inset_xpos + w/2, [te_means[i] for i in inset_idx], w,
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

    add_rank_table(ax, res, ks_avail, panel_table_ks)

fig2.suptitle(
    f'GPT rank sweep  |  {N_FOLDS}-fold CV  |  '
    'phases: {0, pi/2, pi, 3pi/2}^4 = 256 patterns  |  Dashed: chi2/pt=1',
    fontsize=13,
)
rank_plot = os.path.join(args.outdir, "rank_selection_all.png")
fig2.savefig(rank_plot, dpi=150, bbox_inches='tight')
print(f"Rank plot saved -> {rank_plot}")
