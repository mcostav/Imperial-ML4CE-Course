
import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import time
import sobol_seq
from scipy.optimize import minimize
import math

# ---------------------------
# Objective wrapper (provided)
# ---------------------------
def objective_func(X: list):
    return np.array(virtual_lab.conduct_experiment(X))  # shape (n,) or (n, M)

# ---------------------------
# Utilities: encoding & safety
# ---------------------------
CELL_MAP = {'celltype_1': 0, 'celltype_2': 1, 'celltype_3': 2}

def canonical_tuple(x):
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), str(x[5]))

def encode_points(points):
    """
    points: list of [t, pH, f1, f2, f3, 'celltype_*']
    Returns:
      X_cont (n,5) floats, X_cat (n,) ints
    """
    n = len(points)
    Xc = np.zeros((n, 5), dtype=float)
    Xcat = np.zeros(n, dtype=int)
    for i, x in enumerate(points):
        Xc[i, :] = np.array(x[:5], dtype=float)
        Xcat[i] = CELL_MAP[str(x[5])]
    return Xc, Xcat

# ---------------------------
# Kernels
# ---------------------------
def pairwise_sq_dists(A, B):
    A2 = np.sum(A*A, axis=1)[:, None]
    B2 = np.sum(B*B, axis=1)[None, :]
    return A2 + B2 - 2.0 * (A @ B.T)

def rbf_kernel_ard(X1, X2, lengthscales, sigma_f):
    L = np.asarray(lengthscales, float)
    X1s = X1 / L
    X2s = X2 / L
    D2 = pairwise_sq_dists(X1s, X2s)
    return (sigma_f**2) * np.exp(-0.5 * D2)

def categorical_kernel(X1_cat, X2_cat, sigma_cat):
    return (sigma_cat**2) * (X1_cat[:, None] == X2_cat[None, :]).astype(float)

# ---------------------------
# Explicit GP model (single-output)
# ---------------------------
class ExplicitGP:
    def __init__(self):
        # hypers
        self.lengthscales = None  # (5,)
        self.sigma_f = None
        self.sigma_cat = None
        self.sigma_n = None

        # training data
        self.Xc = None
        self.Xcat = None
        self.y = None
        self.y_mean = 0.0

        # normalization for Xc
        self.Xc_mean = None
        self.Xc_std = None

        # caches
        self.L = None
        self.alpha = None

    def _build_K(self, Xc, Xcat, jitter=1e-8):
        Krbf = rbf_kernel_ard(Xc, Xc, self.lengthscales, self.sigma_f)
        Kcat = categorical_kernel(Xcat, Xcat, self.sigma_cat)
        K = Krbf + Kcat + (self.sigma_n**2) * np.eye(Xc.shape[0]) + jitter * np.eye(Xc.shape[0])
        return K

    def _neg_lml(self, theta_log):
        d = 5
        self.lengthscales = np.exp(theta_log[:d])
        self.sigma_f = np.exp(theta_log[d])
        self.sigma_cat = np.exp(theta_log[d+1])
        self.sigma_n = np.exp(theta_log[d+2])

        try:
            K = self._build_K(self.Xc, self.Xcat)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            lml = -0.5 * (self.y @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(self.y) * math.log(2*math.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e12  # large penalty on failure

    def fit(self, X_list, y, optimize_hypers=True, n_restarts=5, random_state=0):
        # encode
        Xc_raw, Xcat = encode_points(X_list)
        # normalize X (continuous)
        self.Xc_mean = np.mean(Xc_raw, axis=0)
        self.Xc_std = np.std(Xc_raw, axis=0)
        self.Xc_std[self.Xc_std == 0.0] = 1.0
        Xc = (Xc_raw - self.Xc_mean) / self.Xc_std

        # center y
        y = np.asarray(y, float).reshape(-1)
        self.y_mean = float(y.mean())
        yc = y - self.y_mean

        self.Xc = Xc
        self.Xcat = Xcat
        self.y = yc

        # initial hypers
        rng = np.random.RandomState(random_state)
        ranges = np.ptp(Xc_raw, axis=0)
        init_ll = np.clip(ranges/2.0, 1e-3, 1e3)
        init_sigma_f = np.std(y) if np.std(y) > 0 else 1.0
        init_sigma_cat = init_sigma_f
        init_sigma_n = max(1e-3, 0.1 * init_sigma_f)
        theta0 = np.log(np.concatenate([init_ll, [init_sigma_f, init_sigma_cat, init_sigma_n]]))

        best = (theta0, np.inf)
        if optimize_hypers:
            bounds = [(math.log(1e-3), math.log(1e3))]*5 + \
                     [(math.log(1e-4), math.log(1e3)), (math.log(1e-6), math.log(1e3)), (math.log(1e-6), math.log(1.0))]
            for r in range(n_restarts):
                start = theta0 + rng.normal(scale=0.5, size=theta0.shape)
                res = minimize(self._neg_lml, start, method='L-BFGS-B', bounds=bounds)
                if res.fun < best[1]:
                    best = (res.x, res.fun)
            _ = self._neg_lml(best[0])  # sets final hypers
        else:
            _ = self._neg_lml(theta0)    # heuristic set

        # final factorization & alpha
        K = self._build_K(self.Xc, self.Xcat)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def _kx_train(self, Xc_star, Xcat_star):
        return rbf_kernel_ard(Xc_star, self.Xc, self.lengthscales, self.sigma_f) + \
               categorical_kernel(Xcat_star, self.Xcat, self.sigma_cat)

    def predict(self, X_list, return_var=True):
        Xc_raw, Xcat = encode_points(X_list)
        Xc_star = (Xc_raw - self.Xc_mean) / self.Xc_std
        K_star = self._kx_train(Xc_star, Xcat)
        mean = K_star @ self.alpha + self.y_mean
        if not return_var:
            return mean
        # predictive variance: k(x,x) - k_*^T K^{-1} k_*
        kxx = (self.sigma_f**2) + (self.sigma_cat**2)
        # solve K v = k_*^T
        V = np.linalg.solve(self.L.T, np.linalg.solve(self.L, K_star.T))
        var = np.maximum(1e-12, kxx - np.sum(K_star.T * V, axis=0))
        return mean, var

    def refit_with_same_hypers(self, X_list, y):
        """
        Shadow GP: reuse hypers, recompute normalization and factors on provided data.
        """
        gp = ExplicitGP()
        gp.lengthscales = self.lengthscales.copy()
        gp.sigma_f = self.sigma_f
        gp.sigma_cat = self.sigma_cat
        gp.sigma_n = self.sigma_n

        Xc_raw, Xcat = encode_points(X_list)
        gp.Xc_mean = np.mean(Xc_raw, axis=0)
        gp.Xc_std = np.std(Xc_raw, axis=0)
        gp.Xc_std[gp.Xc_std == 0.0] = 1.0
        gp.Xc = (Xc_raw - gp.Xc_mean) / gp.Xc_std
        gp.Xcat = Xcat

        y = np.asarray(y, float).reshape(-1)
        gp.y_mean = float(y.mean())
        gp.y = y - gp.y_mean

        K = gp._build_K(gp.Xc, gp.Xcat)
        gp.L = np.linalg.cholesky(K)
        gp.alpha = np.linalg.solve(gp.L.T, np.linalg.solve(gp.L, gp.y))
        return gp

# ---------------------------
# Acquisition: Expected Improvement (maximization)
# ---------------------------
def expected_improvement(mean, var, best_y):
    s = np.sqrt(np.maximum(0.0, var))
    imp = mean - best_y
    z = np.zeros_like(mean)
    nz = s > 0
    z[nz] = imp[nz] / s[nz]
    # Standard normal CDF via math.erf (vectorized)
    Phi = 0.5 * (1.0 + np.array([math.erf(val / math.sqrt(2.0)) for val in z]))
    # Standard normal PDF
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z**2)
    EI = imp * Phi + s * phi
    EI[~nz] = 0.0
    return EI

# ---------------------------
# Batch selection: Kriging Believer (greedy q-EI)
# ---------------------------
def select_batch_kriging_believer(gp, candidates, evaluated_set, batch_size, X_train, y_train):
    """
    gp: fitted ExplicitGP (single-output)
    candidates: list of all candidate points
    evaluated_set: set of canonical tuples already evaluated
    X_train, y_train: current dataset (lists/arrays)
    Returns: list of selected points
    """
    # Filter out already evaluated
    remaining = [x for x in candidates if canonical_tuple(x) not in evaluated_set]
    if len(remaining) == 0:
        return []

    selected = []
    shadow = gp.refit_with_same_hypers(X_train, y_train)

    for _ in range(batch_size):
        if len(remaining) == 0:
            break
        # EI on remaining under shadow GP
        m, v = shadow.predict(remaining, return_var=True)
        best_so_far = float(np.max(y_train))
        EI = expected_improvement(m, v, best_so_far)

        idx = int(np.argmax(EI))
        x_star = remaining.pop(idx)
        selected.append(x_star)

        # Kriging Believer: fantasize mean as observation
        y_star = m[idx]
        X_train = X_train + [x_star]
        y_train = np.concatenate([np.asarray(y_train), np.array([y_star])])

        # Update shadow GP quickly (reuse hypers)
        shadow = gp.refit_with_same_hypers(X_train, y_train)

    return selected

# ---------------------------
# BO loop with GP + batch EI (KB)
# ---------------------------
class BO:
    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func,
                 y_index=None, reopt_every=1, random_state=0):
        start_time = datetime.timestamp(datetime.now())

        self.X_initial = X_initial
        # Deduplicate search space
        seen = set()
        self.X_searchspace = []
        for x in X_searchspace:
            t = canonical_tuple(x)
            if t not in seen:
                seen.add(t)
                self.X_searchspace.append(x)

        self.iterations = iterations
        self.batch = batch
        self.y_index = y_index
        self.reopt_every = reopt_every
        self.rng = np.random.RandomState(random_state)

        # Evaluate initial points
        self.X = list(self.X_initial)
        Y0 = objective_func(self.X_initial)
        if Y0.ndim == 2:
            if y_index is None:
                raise ValueError("objective_func returned multi-output (n, M). Please set y_index (0..M-1).")
            self.Y = Y0[:, y_index].reshape(-1)
        else:
            self.Y = Y0.reshape(-1)

        self.time = [datetime.timestamp(datetime.now()) - start_time] * len(self.Y)
        self.evaluated_set = set(canonical_tuple(x) for x in self.X)

        # Fit initial GP
        self.gp = ExplicitGP()
        self.gp.fit(self.X, self.Y, optimize_hypers=True, n_restarts=5, random_state=random_state)

        # Logs
        self.best_values = [float(np.max(self.Y))]
        self.best_points = [self.X[int(np.argmax(self.Y))]]
        self.best_times = [self.time[int(np.argmax(self.Y))]]

        for iteration in range(iterations):
            # Build candidate list excluding already evaluated
            candidates = [x for x in self.X_searchspace if canonical_tuple(x) not in self.evaluated_set]
            if len(candidates) == 0:
                print(f"[Iter {iteration+1}/{self.iterations}] No remaining candidates. Stopping.")
                break

            # Select batch via EI + Kriging Believer
            batch_points = select_batch_kriging_believer(
                self.gp, candidates, self.evaluated_set, self.batch,
                X_train=list(self.X), y_train=np.asarray(self.Y)
            )
            if len(batch_points) == 0:
                # Fallback: pick random unevaluated points
                batch_points = self.rng.choice(candidates, size=min(self.batch, len(candidates)), replace=False).tolist()

            # Evaluate the batch
            Y_batch_all = objective_func(batch_points)
            if Y_batch_all.ndim == 2:
                Y_batch = Y_batch_all[:, self.y_index].reshape(-1)
            else:
                Y_batch = Y_batch_all.reshape(-1)

            # Update data
            self.X.extend(batch_points)
            self.Y = np.concatenate([self.Y, Y_batch])
            self.time += [datetime.timestamp(datetime.now()) - start_time] * len(Y_batch)
            for x in batch_points:
                self.evaluated_set.add(canonical_tuple(x))

            # Update best trackers
            best_idx = int(np.argmax(self.Y))
            self.best_values.append(float(self.Y[best_idx]))
            self.best_points.append(self.X[best_idx])
            self.best_times.append(self.time[best_idx])

            # Refit GP (optionally re-optimize hypers)
            if ((iteration + 1) % self.reopt_every) == 0:
                self.gp.fit(self.X, self.Y, optimize_hypers=True, n_restarts=3, random_state=iteration+1)

            print(f"[Iter {iteration+1}/{self.iterations}] Best so far: {self.best_values[-1]:.4f}")

    def plot_best_over_time(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.best_times, self.best_values, marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel("Best objective")
        plt.title("Best-so-far vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



X_initial = ([[33, 6.25, 10, 20, 20, 'celltype_1'],
              [38, 8,    20, 10, 20, 'celltype_3'],
              [37, 6.8,   0, 50,  0, 'celltype_1'],
              [36, 6.0,   20, 20, 10, 'celltype_3'],
              [36, 6.1,   20, 20, 10, 'celltype_2'],
              [38, 6.0,   30, 50, 10, 'celltype_1']])

temp = np.linspace(30, 40, 5)
pH = np.linspace(6, 8, 5)
f1 = np.linspace(0, 50, 5)
f2 = np.linspace(0, 50, 5)
f3 = np.linspace(0, 50, 5)
celltype = ['celltype_1', 'celltype_2', 'celltype_3']

X_searchspace = [[a,b,c,d,e,f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]

# If your objective returns multiple outputs, set y_index accordingly (e.g., 0)
BO_m = BO(X_initial, X_searchspace, iterations=15, batch=5, objective_func=objective_func,
          y_index=None,  # set to 0..M-1 if objective_func returns (n, M)
          reopt_every=1, random_state=123)

# Optional: plot
BO_m.plot_best_over_time()
