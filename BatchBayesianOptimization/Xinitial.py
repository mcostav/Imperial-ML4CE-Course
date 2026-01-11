import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import time
import sobol_seq
import scipy
from scipy.optimize import minimize

# Group Submission
group_names = ["Marta Garcia Belza", "Eric Lun"]
cid_numbers = ["", ""]
oral_assignement = [1] # 1 for yes to assessment in oral presentation

#Objective function
def objective_func(X: list): 
    return np.array(virtual_lab.conduct_experiment(X))

#Sobol searchspace generation
def sobol_searchspace(
    temp_range=(30.0, 40.0),
    pH_range=(6.0, 8.0),
    f1_range=(0.0, 50.0),
    f2_range=(0.0, 50.0),
    f3_range=(0.0, 50.0),
    celltypes=('celltype_1', 'celltype_2', 'celltype_3'),
    n_samples=None,
    m=None,
    balance_celltypes=False
):
    """
    Generate Sobol-sampled points for:
        [temp, pH, f1, f2, f3, celltype_idx]
    Returns a fully numeric NumPy array (float + int), and a mapping dict.
    """
    # Validate ranges to see if they are a list or tuple of (min, max) and min <= max
    def _check_range(rng, name):
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2 and rng[0] <= rng[1]):
            raise ValueError(f"{name}_range must be (min, max). Got {rng}")
    for name, rng in zip(["temp","pH","f1","f2","f3"], [temp_range,pH_range,f1_range,f2_range,f3_range]):
        _check_range(rng, name)

    celltypes = list(celltypes)
    n_cat = len(celltypes)

    # Determine sample count
    if n_samples is None and m is None:
        n_samples = 5**5 * n_cat  # match original grid count (9375)
    elif n_samples is None and m is not None:
        n_samples = 2**m

    # Generate Sobol points
    U = sobol_seq.i4_sobol_generate(dim_num=6, n=n_samples)

    # Scale continuous variables
    def _scale(u, lo, hi): return lo + u * (hi - lo)
    temp = _scale(U[:, 0], *temp_range)
    pH   = _scale(U[:, 1], *pH_range)
    f1   = _scale(U[:, 2], *f1_range)
    f2   = _scale(U[:, 3], *f2_range)
    f3   = _scale(U[:, 4], *f3_range)

    # Scale of discrete variable (celltype)
    if balance_celltypes:
        cat_idx = np.arange(n_samples) % n_cat  # ensures balanced representation [0, 1, 2, 0, 1, 2, ...]
    else:
        cat_idx = np.minimum((U[:, 5] * n_cat).astype(np.int8), n_cat - 1) # quasi-random assignment (i.e. Sobol)
    
    #Transform to celltype labels from [0,1,2] when needed
    celltype_col = [celltypes[i] for i in cat_idx]

    # Return list of lists
    return [[temp[i], pH[i], f1[i], f2[i], f3[i], celltype_col[i]] for i in range(n_samples)]

    # Return numpy array by stacking numeric columns (float64 for first 5, int8 for last)
    #return np.column_stack([temp, pH, f1, f2, f3, cat_idx])

start_time = datetime.timestamp(datetime.now())
print("Starting evaluation at:", datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"))
X_searchspace = sobol_searchspace(n_samples=270) #List of lists
print("Time after generating search space (s):", datetime.timestamp(datetime.now()) - start_time)
#Evaluate objective function on all points in search space
obj = np.asarray(objective_func(X_searchspace), dtype=float).ravel()
n = obj.size
print("Time after evaluating objective function (s):", datetime.timestamp(datetime.now()) - start_time)
# ---- Top 3 (largest) ----
order_desc = np.argsort(obj)[::-1]   # indices sorted by descending objective
top3_idx = order_desc[:min(3, n)]

print("Best objective:", obj[top3_idx[0]])
print("At point:", X_searchspace[int(top3_idx[0])])
if top3_idx.size >= 2:
    print("Second best objective:", obj[top3_idx[1]])
    print("At point:", X_searchspace[int(top3_idx[1])])
if top3_idx.size >= 3:
    print("Third best objective:", obj[top3_idx[2]])
    print("At point:", X_searchspace[int(top3_idx[2])])

# ---- Two mid points (around median ranks in ascending order) ----
order_asc = order_desc[::-1]               # reuse: ascending order
mid_left_rank  = (n - 1) // 2
mid_right_rank = n // 2
mid_idx = np.unique([order_asc[mid_left_rank], order_asc[mid_right_rank]])

if mid_idx.size == 2:
    print("Mid points objectives:", obj[mid_idx[0]], "and", obj[mid_idx[1]])
    print("At points:", X_searchspace[int(mid_idx[0])], "and", X_searchspace[int(mid_idx[1])])
else:
    print("Mid point objective:", obj[mid_idx[0]])
    print("At point:", X_searchspace[int(mid_idx[0])])

# ---- Worst (smallest) ----
worst_idx = order_asc[0]                   # or np.argmin(obj)
print("Worst objective:", obj[worst_idx])
print("At point:", X_searchspace[int(worst_idx)])

print("Total evaluation time (s):", datetime.timestamp(datetime.now()) - start_time)