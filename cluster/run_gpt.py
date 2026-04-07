"""
Single (group_id, k, fold) optimization job.
Usage: python run_gpt.py --group_id 0 --k 3 --fold 2 --outdir results/
"""
import argparse, os, time
import numpy as np
from itertools import product as _prod
from scipy.optimize import minimize

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--group_id", type=int, required=True, choices=[0,1,2,3,4])
parser.add_argument("--k",        type=int, required=True)
parser.add_argument("--fold",     type=int, required=True)
parser.add_argument("--outdir",   type=str, default="results")
args = parser.parse_args()

group_id = args.group_id
k        = args.k
fold     = args.fold
os.makedirs(args.outdir, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SIGMA_FLOOR = 1e-3
KX_LIST     = [-20, -10, 10, 20]
PHASES      = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 4 phase angles
BEAM_RADIUS = 0.5                            # experimental-mimicking
SLIT_X      = np.array([-0.05, -0.015, 0.015, 0.05])
NUM_PIXELS  = 150
N_FOLDS     = 8
N_RESTARTS  = 8
MAX_ITER    = 5000
TOL         = 1e-3

x_grid = np.linspace(-1, 1, NUM_PIXELS)

# Shutter patterns in identifiers.txt order
SHUTTER_PATTERNS = [
    'XXXX', 'XXOX', 'XXXO', 'XXOO', 'OXXX', 'OXOX', 'OXXO', 'OXOO',
    'XOXX', 'XOOX', 'XOXO', 'XOOO', 'OOXX', 'OOOX', 'OOXO', 'OOOO'
]
slits_open_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 'all'}
n_slits_open   = slits_open_map[group_id]

# ── Simulate ───────────────────────────────────────────────────────────────────
rows = []
for sh in SHUTTER_PATTERNS:
    open_slits = tuple(i for i, ch in enumerate(sh) if ch == 'O')
    if n_slits_open == 'all':
        if len(open_slits) == 0:
            continue
    elif len(open_slits) != n_slits_open:
        continue
    for phases in _prod(PHASES, repeat=4):
        field = sum(
            np.exp(-2 * (x_grid - SLIT_X[s])**2 / BEAM_RADIUS**2)
            * np.exp(1j * (phases[s] + KX_LIST[s] * x_grid))
            for s in open_slits
        )
        rows.append(np.abs(field)**2)

raw = np.array(rows)
num_pixels = NUM_PIXELS
n_s = raw.shape[0]
print(f"group={group_id}  slits_open={n_slits_open}  n_s={n_s}  k={k}  fold={fold}")

# ── Normalise ──────────────────────────────────────────────────────────────────
row_max = raw.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
data  = raw / row_max
sigma = np.sqrt(data + SIGMA_FLOOR**2)

# ── CV fold masks ──────────────────────────────────────────────────────────────
n_total   = n_s * num_pixels
perm      = np.random.default_rng(42).permutation(n_total)
fold_ids  = perm % N_FOLDS
test_mask  = (fold_ids.reshape(n_s, num_pixels) == fold).astype(float)
train_mask = 1 - test_mask

# ── GPT model: Î_s(x) = omega · T_s · X_x · e ────────────────────────────────
def unpack(xv):
    c     = 0
    omega = np.append(1.0, xv[c:c + k - 1]);  c += k - 1
    T     = [xv[c + i*k**2 : c + (i+1)*k**2].reshape(k, k) for i in range(n_s)]
    c    += n_s * k**2
    X_all = xv[c:c + num_pixels * k**2].reshape(num_pixels, k, k);  c += num_pixels * k**2
    e     = xv[c:c + k]
    return omega, T, X_all, e

def predict(xv):
    omega, T, X_all, e = unpack(xv)
    rows = []
    for s in range(n_s):
        v_s = omega @ T[s]
        rows.append(np.einsum('i,xij,j->x', v_s, X_all, e))
    return np.vstack(rows)

def chi2(xv, train_mask):
    return np.sum((predict(xv) - data)**2 / sigma**2 * train_mask)

def make_x0(seed=42):
    rng = np.random.default_rng(seed)
    eps = 0.05
    x0  = list(rng.uniform(-eps, eps, k - 1))
    for _ in range(n_s):
        x0 += list((np.eye(k) + rng.uniform(-eps, eps, (k, k))).flatten())
    X_init = np.zeros((num_pixels, k, k))
    for px in range(num_pixels):
        X_init[px] = np.eye(k) + rng.uniform(-eps, eps, (k, k))
    x0 += list(X_init.flatten())
    e_init    = rng.uniform(-eps, eps, k)
    e_init[0] = data.mean()
    x0 += list(e_init)
    return np.array(x0)

# ── Optimise ───────────────────────────────────────────────────────────────────
constraints = (
    {'type': 'ineq', 'fun': lambda xv: np.min(predict(xv))},
    {'type': 'ineq', 'fun': lambda xv: 1.0 - np.max(predict(xv))},
)

best_res = None
t0 = time.time()
for restart in range(N_RESTARTS):
    x0  = make_x0(seed=42 + fold * N_RESTARTS + restart)
    res = minimize(chi2, x0,
                   args=(train_mask,),
                   method='SLSQP',
                   constraints=constraints,
                   options={'maxiter': MAX_ITER},
                   tol=TOL)
    if best_res is None or res.fun < best_res.fun:
        best_res = res
    print(f"  restart {restart}  loss={res.fun:.5f}  {'OK' if res.success else 'NOT converged'}")

elapsed = time.time() - t0
res = best_res

pred      = predict(res.x)
train_err = np.sum((pred - data)**2 / sigma**2 * train_mask) / train_mask.sum()
test_err  = np.sum((pred - data)**2 / sigma**2 * test_mask)  / test_mask.sum()

print(f"  fold {fold}  train={train_err:.5f}  test={test_err:.5f}"
      f"  time={elapsed:.1f}s  {'OK' if res.success else 'NOT converged'}")

# ── Save ───────────────────────────────────────────────────────────────────────
outfile = os.path.join(args.outdir, f"result_group{group_id}_k{k}_fold{fold}.npz")
np.savez(outfile,
         x_opt      = res.x,
         train_err  = train_err,
         test_err   = test_err,
         elapsed    = elapsed,
         success    = res.success,
         group_id   = group_id,
         k          = k,
         fold       = fold,
         n_s        = n_s,
         num_pixels = num_pixels,
         sigma_floor= SIGMA_FLOOR)
print(f"Saved → {outfile}")
