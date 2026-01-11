
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------
# Bounds per variable
B_TEMP = (30.0, 40.0)
B_PH   = (6.0, 8.0)
B_F1   = (0.0, 50.0)
B_F2   = (0.0, 50.0)
B_F3   = (0.0, 50.0)

# Exactly 5 levels per continuous variable (max 5 as requested)
N_LEVELS = 5

# Categorical variable
CELLTYPES = ['celltype_1', 'celltype_2', 'celltype_3']

# Number of Sobol points to generate BEFORE snapping to levels.
# Tip: use a power of two (e.g., 1024, 2048, 4096) for best Sobol balance.
# 3125 (5**5) is the max number of unique continuous-level tuples,
# but Sobol prefers powers of two; 2048 is a good starting choice.
N_target = 2048

# Optional seed for reproducibility
SEED = 42

# ------------------------------------------------------------
# 2) Build discrete levels and helper functions
# ------------------------------------------------------------
def make_levels(bounds, n_levels):
    lo, hi = bounds
    return np.linspace(lo, hi, n_levels)

TEMP_LEVELS = make_levels(B_TEMP, N_LEVELS)
PH_LEVELS   = make_levels(B_PH,   N_LEVELS)
F1_LEVELS   = make_levels(B_F1,   N_LEVELS)
F2_LEVELS   = make_levels(B_F2,   N_LEVELS)
F3_LEVELS   = make_levels(B_F3,   N_LEVELS)

# For snapping continuous values to the nearest of the 5 levels
def snap_to_levels(values, levels):
    # values: (n,) continuous
    # levels: (5,) sorted
    # returns index in {0..4}
    idx = np.searchsorted(levels, values, side='left')
    # Correct indices that hit the right boundary
    idx = np.clip(idx, 0, len(levels)-1)
    # Compare left vs right neighbor to choose the nearest
    left_is_better = (idx > 0) & (
        np.abs(values - levels[idx-1]) <= np.abs(values - levels[idx])
    )
    idx[left_is_better] -= 1
    return idx  # integer indices 0..4

# ------------------------------------------------------------
# 3) Generate Sobol points in the continuous hyper-rectangle
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)

# 5D (temp, pH, f1, f2, f3)
bounds_arr = np.array([
    [B_TEMP[0], B_TEMP[1]],
    [B_PH[0],   B_PH[1]],
    [B_F1[0],   B_F1[1]],
    [B_F2[0],   B_F2[1]],
    [B_F3[0],   B_F3[1]],
], dtype=float)

sampler = qmc.Sobol(d=5, scramble=True, seed=SEED)
# Sobol prefers n as a power of 2; you can set N_target accordingly.
X_unit = sampler.random(N_target)  # in [0,1]^5
X_cont = qmc.scale(X_unit, bounds_arr[:, 0], bounds_arr[:, 1])  # shape (N, 5)

# ------------------------------------------------------------
# 4) Snap Sobol points to the 5 discrete levels per variable
# ------------------------------------------------------------
i_temp = snap_to_levels(X_cont[:, 0], TEMP_LEVELS)
i_pH   = snap_to_levels(X_cont[:, 1], PH_LEVELS)
i_f1   = snap_to_levels(X_cont[:, 2], F1_LEVELS)
i_f2   = snap_to_levels(X_cont[:, 3], F2_LEVELS)
i_f3   = snap_to_levels(X_cont[:, 4], F3_LEVELS)

# Combine level indices -> unique discrete tuples (continuous part)
idx_tuples = np.stack([i_temp, i_pH, i_f1, i_f2, i_f3], axis=1)  # (N, 5)
# Keep unique combinations in the order they first appear
# (np.unique would sort; we preserve sequence with a dict)
seen = {}
order = []
for row in idx_tuples:
    key = tuple(row.tolist())
    if key not in seen:
        seen[key] = True
        order.append(key)
idx_unique = np.array(order, dtype=int)  # shape (M, 5), M <= 3125

# Map indices back to actual level values
X_cont_discrete = np.column_stack([
    TEMP_LEVELS[idx_unique[:, 0]],
    PH_LEVELS[idx_unique[:, 1]],
    F1_LEVELS[idx_unique[:, 2]],
    F2_LEVELS[idx_unique[:, 3]],
    F3_LEVELS[idx_unique[:, 4]],
])

# ------------------------------------------------------------
# 5) Attach categorical celltype as evenly as possible
# ------------------------------------------------------------
M = X_cont_discrete.shape[0]
# Balanced assignment across 3 categories
reps = M // len(CELLTYPES)
rem  = M %  len(CELLTYPES)
cell_assign = []
for ct in CELLTYPES:
    cell_assign += [ct] * reps
# Distribute the remainder
cell_assign += CELLTYPES[:rem]
cell_assign = np.array(cell_assign, dtype=object)

# Shuffle to avoid blocks of the same celltype
shuf_idx = rng.permutation(M)
X_cont_discrete = X_cont_discrete[shuf_idx]
cell_assign = cell_assign[shuf_idx]

# Build final mixed search space list: [temp, pH, f1, f2, f3, celltype]
X_searchspace_sobol = [
    [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), cell_assign[i]]
    for i, row in enumerate(X_cont_discrete)
]

print(f"Total unique Sobol-discretized points (continuous part): {M}")
print(f"Final mixed points with celltype attached: {len(X_searchspace_sobol)}")

# ------------------------------------------------------------
# 6) Visual checks / plots
# ------------------------------------------------------------
plt.style.use('seaborn-v0_8')

# a) Counts per discrete level (0..4) for each variable
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes = axes.ravel()
labels = ['Temp', 'pH', 'f1', 'f2', 'f3']
level_arrays = [
    idx_unique[:, 0], idx_unique[:, 1], idx_unique[:, 2], idx_unique[:, 3], idx_unique[:, 4]
]

for i, (lab, arr) in enumerate(zip(labels, level_arrays)):
    counts = np.bincount(arr, minlength=N_LEVELS)
    axes[i].bar(np.arange(N_LEVELS), counts, color='#1f77b4')
    axes[i].set_xticks(np.arange(N_LEVELS))
    axes[i].set_xlabel(f'{lab} level index (0..{N_LEVELS-1})')
    axes[i].set_ylabel('Count')
    axes[i].set_title(f'{lab}: counts per level')
fig.delaxes(axes[5])
fig.tight_layout()
plt.show()

# b) 2D scatter (Temp vs pH)
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(X_cont_discrete[:, 0], X_cont_discrete[:, 1], s=14, alpha=0.7, c='#2ca02c')
ax2.set_xlabel('Temp (°C)')
ax2.set_ylabel('pH')
ax2.set_title('Temp vs pH (Sobol-discretized)')
ax2.grid(True, alpha=0.2)
plt.show()

# c) 3D scatter (f1, f2, f3) colored by celltype
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
colors_map = {'celltype_1': '#1f77b4', 'celltype_2': '#ff7f0e', 'celltype_3': '#2ca02c'}
for ct in CELLTYPES:
    mask = (cell_assign == ct)
    ax3.scatter(
        X_cont_discrete[mask, 2],  # f1
        X_cont_discrete[mask, 3],  # f2
        X_cont_discrete[mask, 4],  # f3
        s=12, alpha=0.75, c=colors_map[ct], label=ct
    )
ax3.set_xlabel('f1'); ax3.set_ylabel('f2'); ax3.set_zlabel('f3')
ax3.set_title('f1–f2–f3 (Sobol-discretized), colored by celltype')
ax3.legend()
plt.show()

# d) Celltype counts
fig4, ax4 = plt.subplots(figsize=(6, 4))
counts_ct = [np.sum(cell_assign == ct) for ct in CELLTYPES]
ax4.bar(CELLTYPES, counts_ct, color=[colors_map[c] for c in CELLTYPES])
ax4.set_ylabel('Count')
ax4.set_title('Celltype counts')
for i, v in enumerate(counts_ct):
    ax4.text(i, v, str(v), ha='center', va='bottom')
plt.show()

# ------------------------------------------------------------
# 7) Optional: compare with your brute grid definition
# ------------------------------------------------------------
# If you want to keep your original exact 5-level grid in a list (huge!), you can still do:
# temp = TEMP_LEVELS
# pH   = PH_LEVELS
# f1   = F1_LEVELS
# f2   = F2_LEVELS
# f3   = F3_LEVELS
# celltype = CELLTYPES
# X_searchspace_grid = [[a, b, c, d, e, f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]
# This will be length 9375; consider memory/time if you try to plot all of them!
