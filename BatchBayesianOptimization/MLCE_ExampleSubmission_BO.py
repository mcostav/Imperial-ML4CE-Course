import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import time
import sobol_seq
import scipy

# Group Submission
group_names = ["Marta Garcia Belza", "Eric Lun"]
cid_numbers = ["", ""]
oral_assignement = [1]

#Objective function
def objective_func(X: list): 
    return(np.array(virtual_lab.conduct_experiment(X)))

def acquisition_EI(mean, var, best_y):
    """ Expected Improvement for maximization """
    s = np.sqrt(np.maximum(0.0, var))
    improvement = mean - best_y
    z = np.zeros_like(mean)
    nz = s > 0
    z[nz] = improvement[nz] / s[nz]

    # CDF and PDF of standard normal
    Phi = 0.5 * (1.0 + scipy.special.erf(z / np.sqrt(2.0)))
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

    EI = improvement * Phi + s * phi
    EI[~nz] = 0.0
    return EI

#Helper class - Gaussian Process
class GP:
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        
        #GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim = X.shape[0], X.shape[1]
        self.ny_dim = Y.shape[1]
        self.multi_hyper = multi_hyper
        self.var_out = var_out

        #normalize data
        self.X_mean, self.X_std = np.mean(self.X, axis=0), np.std(self.X, axis=0)
        self.Y_mean, self.Y_std = np.mean(self.Y, axis=0), np.std(self.Y, axis=0)
        self.X_norm, self.Y_norm = (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

        # Determine hyperparameters
        self.hypopt, self.invKopt = self.determine_hyperparameters()

    def Cov_mat(self, kernel, X_norm, W, sf2):
        """ Compute covariance matrix of a dataset Xnorm """
        if kernel == 'RBF':
            dist = cdist(X_norm, X_norm, 'seuclidean', V=W)**2
            cov_matrix = sf2 * np.exp(-0.5 * dist)
            return cov_matrix
        else:
            print('ERROR no kernel with name', kernel)

    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        '''Calculates the covariance of a single sample xnorm against the dataset Xnorm'''
        # Internal parameters
        nx_dim = self.nx_dim
        dist = cdist(Xnorm, xnorm.reshape(1, nx_dim), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-0.5 * dist)
        return cov_matrix
    
    def negative_log_likelihood(self, hyper, X, Y):
        #internal parameters
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel = self.kernel

        W = np.exp(2*hyper[:nx_dim])
        sf2 = np.exp(2*hyper[nx_dim])
        sn2 = np.exp(2*hyper[nx_dim+1])

        K = self.Cov_mat(kernel, X, W, sf2) + (sn2 + 1e-8) * np.eye(n_point)
        K = (K + K.T) * 0.5  # Ensure symmetry
        L = np.linalg.cholesky(K)
        logdetK = 2.0 * np.sum(np.log(np.diag(L)))
        invLY = np.linalg.solve(L, Y)
        alpha = np.linalg.solve(L.T, invLY)
        NLL = np.dot(Y.T, alpha) + logdetK
        
        return NLL

    def rbf_kernel(self, X1, X2):
        """ RBF kernel with ARD """
        L = np.asarray(self.lengthscales, dtype=float)
        X1s = X1 / L
        X2s = X2 / L
        D2 = np.sum(X1s**2, axis=1)[:, None] + np.sum(X2s**2, axis=1)[None, :] - 2.0 * (X1s @ X2s.T)
        return (self.sigma_f**2) * np.exp(-0.5 * D2)

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)

        # Hyperparameters (fixed for simplicity)
        self.lengthscales = np.std(self.X_train, axis=0) / 2.0 + 1e-3
        self.sigma_f = np.std(self.y_train) + 1e-3
        self.sigma_n = 1e-2 * self.sigma_f + 1e-3

        K = self.rbf_kernel(self.X_train, self.X_train)
        K += (self.sigma_n**2) * np.eye(self.X_train.shape[0]) + 1e-8 * np.eye(self.X_train.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

    def predict(self, X_star):
        Xs = np.asarray(X_star, dtype=float)
        K_star = self.rbf_kernel(Xs, self.X_train)
        mean = K_star @ self.alpha

        v = np.linalg.solve(self.L, K_star.T)
        kxx = (self.sigma_f**2) * np.ones(Xs.shape[0])
        var = np.maximum(1e-12, kxx - np.sum(v**2, axis=0))
        return mean, var

class RandomSelection:
    def __init__(self, X_searchspace, objective_func, batch): 
        self.X_searchspace = X_searchspace
        self.batch = batch

        random_searchspace = [self.X_searchspace[random.randrange(len(self.X_searchspace))] for c in range(batch)]
        self.random_Y = objective_func(random_searchspace)


class BO:
    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func):
        start_time = datetime.timestamp(datetime.now())

        self.X_initial = X_initial
        self.X_searchspace = X_searchspace
        self.iterations = iterations
        self.batch = batch

        self.Y = objective_func(self.X_initial)
        self.time = [datetime.timestamp(datetime.now())-start_time]*(len(self.Y))
        
        for iteration in range(iterations):
            random_selection = RandomSelection(self.X_searchspace, objective_func, self.batch)
            print(f"[Iter {iteration+1}/{self.iterations}] Best so far: {np.max(self.Y):.4f}")
            self.Y = np.concatenate([self.Y, random_selection.random_Y])
            self.time += [datetime.timestamp(datetime.now())-start_time]*(len(random_selection.random_Y))


X_initial = ([[33, 6.25, 10, 20, 20, 'celltype_1'],
              [38, 8, 20, 10, 20, 'celltype_3'],
              [37, 6.8, 0, 50, 0, 'celltype_1'],
              [36, 6.0, 20, 20, 10, 'celltype_3'],
              [36, 6.1, 20, 20, 10, 'celltype_2'],
              [38, 6.0, 30, 50, 10, 'celltype_1']])

temp = np.linspace(30,40,5)
pH = np.linspace(6, 8, 5)
f1 = np.linspace(0, 50, 5)
f2 = np.linspace(0, 50,5)
f3 = np.linspace(0, 50,5)
celltype = ['celltype_1','celltype_2','celltype_3']

X_searchspace     = [[a,b,c,d,e,f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]
BO_m = BO(X_initial, X_searchspace, 15, 5, objective_func)


'''
import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import time
import sobol_seq
import scipy
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf

# ---------------------------
# Objective wrapper (provided)
# ---------------------------
def objective_func(X: list):
    # Expects a list of points; returns np.array of shape (len(X),)
    return np.array(virtual_lab.conduct_experiment(X))

# ---------------------------
# Utilities: encoding & kernels
# ---------------------------
CELL_MAP = {'celltype_1': 0, 'celltype_2': 1, 'celltype_3': 2}

def encode_X(X_list):
    """
    Convert list of points [t,pH,f1,f2,f3,'celltype_*'] to:
      X_cont: np.array of shape (n,5)
      X_cat:  np.array of shape (n,)
    """
    n = len(X_list)
    X_cont = np.zeros((n, 5), dtype=float)
    X_cat = np.zeros(n, dtype=int)
    for i, x in enumerate(X_list):
        X_cont[i, :] = np.array(x[:5], dtype=float)
        X_cat[i] = CELL_MAP[x[5]]
    return X_cont, X_cat

def pairwise_sq_dists(A, B):
    """
    Efficient pairwise squared distances for rows of A and B.
    A: (m,d), B: (n,d)
    Returns: (m,n)
    """
    A2 = np.sum(A*A, axis=1)[:, None]
    B2 = np.sum(B*B, axis=1)[None, :]
    return A2 + B2 - 2.0 * (A @ B.T)

def rbf_kernel_ard(X1, X2, lengthscales, sigma_f):
    """
    ARD RBF kernel: k(x,x') = sigma_f^2 * exp(-0.5 * sum_j ((x_j - x'_j)^2 / l_j^2))
    X1: (n1, d), X2: (n2, d)
    lengthscales: (d,), sigma_f: scalar > 0
    Returns: (n1, n2)
    """
    # Scale inputs by lengthscales
    L = np.asarray(lengthscales, dtype=float)
    X1s = X1 / L
    X2s = X2 / L
    D2 = pairwise_sq_dists(X1s, X2s)  # (n1, n2)
    return (sigma_f**2) * np.exp(-0.5 * D2)

def categorical_kernel(X1_cat, X2_cat, sigma_cat):
    """
    Simple categorical kernel: k_cat = sigma_cat^2 if same category else 0
    X1_cat: (n1,), X2_cat: (n2,), integers
    Returns: (n1, n2)
    """
    return (sigma_cat**2) * (X1_cat[:, None] == X2_cat[None, :]).astype(float)

# ---------------------------
# Gaussian Process Regression
# ---------------------------
class ExplicitGP:
    def __init__(self):
        # Hyperparameters (in positive domain)
        self.lengthscales = None  # (5,)
        self.sigma_f = None
        self.sigma_cat = None
        self.sigma_n = None

        # Training data
        self.X_cont = None
        self.X_cat = None
        self.y = None

        # Cached linear algebra
        self.L = None         # Cholesky factor of K
        self.alpha = None     # K^{-1} y via cho_solve

    def _build_K(self, Xc, Xcat, jitter=1e-8):
        Krbf = rbf_kernel_ard(Xc, Xc, self.lengthscales, self.sigma_f)
        Kcat = categorical_kernel(Xcat, Xcat, self.sigma_cat)
        K = Krbf + Kcat
        # Add noise on the diagonal for observations
        K += (self.sigma_n**2) * np.eye(Xc.shape[0])
        # Add tiny jitter for numerical stability
        K += jitter * np.eye(Xc.shape[0])
        return K

    def _neg_log_marginal_likelihood(self, theta_log):
        """
        theta_log: [log l1..l5, log sigma_f, log sigma_cat, log sigma_n]
        Returns negative log marginal likelihood (to be minimized).
        """
        d = 5
        ll = np.exp(theta_log[:d])           # lengthscales
        sigma_f = np.exp(theta_log[d])
        sigma_cat = np.exp(theta_log[d+1])
        sigma_n = np.exp(theta_log[d+2])

        # Temporarily set
        self.lengthscales = ll
        self.sigma_f = sigma_f
        self.sigma_cat = sigma_cat
        self.sigma_n = sigma_n

        try:
            K = self._build_K(self.X_cont, self.X_cat)
            # Cholesky
            L, lower = cho_factor(K, lower=True, check_finite=False)
            alpha = cho_solve((L, lower), self.y, check_finite=False)
            # Log marginal likelihood
            lml = -0.5 * (self.y @ alpha) \
                  - np.sum(np.log(np.diag(L))) \
                  - 0.5 * self.X_cont.shape[0] * np.log(2*np.pi)
            return -lml
        except np.linalg.LinAlgError:
            # If numerical issues, penalize
            return 1e6

    def fit(self, X_cont, X_cat, y, optimize_hypers=True, n_restarts=5, random_state=42):
        """
        Fit GP to data. If optimize_hypers=True, run L-BFGS with restarts.
        """
        self.X_cont = np.asarray(X_cont, dtype=float)
        self.X_cat = np.asarray(X_cat, dtype=int)
        self.y = np.asarray(y, dtype=float)

        # Simple mean-centering to help stability (optional)
        self.y_mean = np.mean(self.y)
        y_center = self.y - self.y_mean
        self.y = y_center

        # Initialize hyperparameters
        rng = np.random.RandomState(random_state)

        # Heuristic initial lengthscales ~ ranges / 2
        ranges = np.ptp(self.X_cont, axis=0)
        init_ll = np.clip(ranges / 2.0, 1e-3, 1e3)  # avoid extremes
        init_sigma_f = np.std(self.y) if np.std(self.y) > 0 else 1.0
        init_sigma_cat = init_sigma_f
        init_sigma_n = max(1e-2, 0.1 * init_sigma_f)

        best_theta = np.log(np.concatenate([init_ll, [init_sigma_f, init_sigma_cat, init_sigma_n]]))
        best_val = np.inf

        if optimize_hypers:
            for r in range(n_restarts):
                # Randomize start in log-space
                noise = rng.normal(scale=0.5, size=best_theta.shape)
                theta0 = best_theta + noise
                # Bounds in log-space: [log(1e-3), log(1e3)] for positives
                bounds = [(np.log(1e-3), np.log(1e3))] * 5 + \
                         [(np.log(1e-4), np.log(1e3)),  # sigma_f
                          (np.log(1e-6), np.log(1e3)),  # sigma_cat
                          (np.log(1e-6), np.log(1e0))]  # sigma_n (noise up to 1)

                res = minimize(self._neg_log_marginal_likelihood,
                               theta0, method='L-BFGS-B', bounds=bounds)
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x

        # Set best hypers
        d = 5
        self.lengthscales = np.exp(best_theta[:d])
        self.sigma_f = np.exp(best_theta[d])
        self.sigma_cat = np.exp(best_theta[d+1])
        self.sigma_n = np.exp(best_theta[d+2])

        # Final factorization
        K = self._build_K(self.X_cont, self.X_cat)
        self.L, self.lower = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve((self.L, self.lower), self.y, check_finite=False)

    def predict(self, X_cont_star, X_cat_star, return_var=True):
        """
        Vectorized GP prediction for many test points.
        Returns mean (M,), var (M,) if return_var True; otherwise mean only.
        """
        Xc_star = np.asarray(X_cont_star, dtype=float)
        Xcat_star = np.asarray(X_cat_star, dtype=int)

        K_star = rbf_kernel_ard(Xc_star, self.X_cont, self.lengthscales, self.sigma_f) \
                 + categorical_kernel(Xcat_star, self.X_cat, self.sigma_cat)
        # Predictive mean (re-add mean)
        mean = K_star @ self.alpha + self.y_mean

        if not return_var:
            return mean

        # Predictive variance: k(x,x) - k_*^T K^{-1} k_*
        # For our kernel, k(x,x) = sigma_f^2 + sigma_cat^2 (no noise term)
        kxx = (self.sigma_f**2) + (self.sigma_cat**2)
        # Solve L v = K_star^T for all test points
        V = cho_solve((self.L, self.lower), K_star.T, check_finite=False)  # shape (N, M)
        var = np.maximum(1e-12, kxx - np.sum(K_star.T * V, axis=0))
        return mean, var

# ---------------------------
# Acquisition: Expected Improvement (maximization)
# ---------------------------
def expected_improvement(mean, var, best_y):
    """
    EI for maximization. mean, var are arrays of same length; best_y is scalar.
    EI(x) = (m - best) * Phi(z) + s * phi(z), with z = (m - best) / s
    """
    s = np.sqrt(np.maximum(0.0, var))
    improvement = mean - best_y
    z = np.zeros_like(mean)
    nz = s > 0
    z[nz] = improvement[nz] / s[nz]

    # Phi(z) and phi(z)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

    EI = improvement * Phi + s * phi
    EI[~nz] = 0.0
    return EI

# ---------------------------
# Batch selection: Kriging Believer
# ---------------------------
def select_batch_kriging_believer(gp, X_searchspace, X_evaluated_set, batch_size):
    """
    Greedy batch selection using EI + Kriging Believer (fantasized mean).
    gp: fitted ExplicitGP (hyperparams fixed during batch selection)
    X_searchspace: list of all candidate points
    X_evaluated_set: set of tuples already evaluated (to avoid duplicates)
    batch_size: int
    Returns: list of selected points
    """
    # Start from current gp state
    Xc_train = gp.X_cont.copy()
    Xcat_train = gp.X_cat.copy()
    y_train = gp.y.copy() + gp.y_mean  # revert to original scale for fantasizing
    # But GP was trained on y_centered; for fantasizing we need consistent behavior.
    # We'll maintain a shadow GP with the same hypers, updating y centered correctly.

    selected = []
    available_mask = np.array([tuple(x) not in X_evaluated_set for x in X_searchspace], dtype=bool)
    candidates = [X_searchspace[i] for i in np.where(available_mask)[0]]

    # Pre-encode all candidates for vectorized predictions
    Xc_cand, Xcat_cand = encode_X(candidates)

    # Shadow GP that we will update with fantasized points without re-optimizing hypers
    shadow = ExplicitGP()
    shadow.lengthscales = gp.lengthscales.copy()
    shadow.sigma_f = gp.sigma_f
    shadow.sigma_cat = gp.sigma_cat
    shadow.sigma_n = gp.sigma_n
    shadow.y_mean = gp.y_mean

    shadow.X_cont = gp.X_cont.copy()
    shadow.X_cat = gp.X_cat.copy()
    shadow.y = gp.y.copy()
    shadow.L = gp.L
    shadow.lower = gp.lower
    shadow.alpha = gp.alpha

    for b in range(batch_size):
        # Predict EI for all available candidates
        mean_cand, var_cand = shadow.predict(Xc_cand, Xcat_cand, return_var=True)
        best_so_far = (shadow.y + shadow.y_mean).max()  # original scale
        EI = expected_improvement(mean_cand, var_cand, best_so_far)

        # If all EI are zero (numerically), fall back to random among available
        if np.all(EI <= 1e-12):
            idx = np.random.choice(len(candidates))
        else:
            idx = int(np.argmax(EI))

        x_star = candidates[idx]
        selected.append(x_star)

        # Remove this candidate from the candidate pool
        candidates.pop(idx)
        Xc_cand = np.delete(Xc_cand, idx, axis=0)
        Xcat_cand = np.delete(Xcat_cand, idx, axis=0)

        # Kriging Believer: fantasize y as GP mean and update shadow GP quickly
        x_star_c, x_star_cat = encode_X([x_star])
        m_star = mean_cand[idx]  # already on original scale

        # Update shadow training set (centered y)
        shadow.X_cont = np.vstack([shadow.X_cont, x_star_c])
        shadow.X_cat = np.concatenate([shadow.X_cat, x_star_cat])
        y_new_centered = np.concatenate([shadow.y, np.array([m_star - shadow.y_mean])])
        shadow.fit(shadow.X_cont, shadow.X_cat, y_new_centered + shadow.y_mean,
                   optimize_hypers=False)  # reuse hypers, update factors only

    return selected

# ---------------------------
# BO loop with GP + batch q-EI (Kriging Believer)
# ---------------------------
class BO:
    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func,
                 optimize_hypers_every=1, random_state=123):
        """
        Bayesian Optimization with batch selection.
        - Maximizes objective by EI.
        - optimize_hypers_every: how often to re-optimize GP hyperparams (in iterations).
        """
        start_time = datetime.timestamp(datetime.now())
        rng = np.random.RandomState(random_state)

        self.X_initial = X_initial
        self.X_searchspace = X_searchspace
        self.iterations = iterations
        self.batch = batch

        # Evaluate initial points
        self.X = list(self.X_initial)  # list of points
        self.Y = objective_func(self.X_initial)  # np.array shape (n0,)
        self.time = [datetime.timestamp(datetime.now()) - start_time] * len(self.Y)

        # Track best
        self.best_y = [np.max(self.Y)]
        self.best_time = [self.time[np.argmax(self.Y)]]
        self.best_x = [self.X[int(np.argmax(self.Y))]]

        # Build evaluated set to avoid duplicates
        self.X_evaluated_set = set([tuple(x) for x in self.X])

        # Fit GP on initial data
        Xc, Xcat = encode_X(self.X)
        gp = ExplicitGP()
        gp.fit(Xc, Xcat, self.Y, optimize_hypers=True, n_restarts=5, random_state=random_state)

        for it in range(self.iterations):
            # Select a batch via EI + Kriging Believer
            batch_points = select_batch_kriging_believer(gp, self.X_searchspace,
                                                         self.X_evaluated_set, self.batch)

            # Evaluate the batch with the true objective function
            Y_batch = objective_func(batch_points)

            # Update main records
            self.X.extend(batch_points)
            self.Y = np.concatenate([self.Y, Y_batch])
            self.time += [datetime.timestamp(datetime.now()) - start_time] * len(Y_batch)

            # Update evaluated set
            for x in batch_points:
                self.X_evaluated_set.add(tuple(x))

            # Update best trackers
            current_best_idx = int(np.argmax(self.Y))
            self.best_y.append(self.Y[current_best_idx])
            self.best_time.append(self.time[current_best_idx])
            self.best_x.append(self.X[current_best_idx])

            # Refit GP with (optionally) re-optimized hyperparameters
            Xc, Xcat = encode_X(self.X)
            reopt = ((it + 1) % optimize_hypers_every == 0)
            gp.fit(Xc, Xcat, self.Y, optimize_hypers=reopt, n_restarts=3, random_state=random_state)

            # Optional: simple progress print
            print(f"[Iter {it+1}/{self.iterations}] Best so far: {self.best_y[-1]:.4f}")

        # Save final GP if you want to inspect later
        self.gp = gp

    def plot_best_over_time(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.best_time, self.best_y, marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel("Best objective (max)")
        plt.title("Best-so-far vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------
# Example usage with your data
# ---------------------------
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

# Run BO: 15 iterations, batch size 5
BO_m = BO(X_initial, X_searchspace, iterations=15, batch=5, objective_func=objective_func,
          optimize_hypers_every=1, random_state=123)

# Optional plot
BO_m.plot_best_over_time()
'''