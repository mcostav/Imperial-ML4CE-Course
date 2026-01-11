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
    balance_celltypes=True
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
    
    #Transform to celltype labels if needed
    #celltype_col = [celltypes[i] for i in cat_idx]

    # Stack numeric columns (float64 for first 5, int8 for last)
    return np.column_stack([temp, pH, f1, f2, f3, cat_idx])


#Objective function
def objective_func(X: list): 
    return(np.array(virtual_lab.conduct_experiment(X)))
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
class GP_model:
    
    ###########################
    # --- initializing GP --- #
    ###########################    
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        
        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim   = X.shape[0], X.shape[1]
        self.ny_dim                 = Y.shape[1]
        self.multi_hyper            = multi_hyper
        self.var_out                = var_out
        
        # normalize data
        self.X_mean, self.X_std     = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std     = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm    = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std
        
        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()        
    
    #############################
    # --- Covariance Matrix --- #
    #############################

    def Cov_mat(self, kernel, X_norm, W, sf2):
        if kernel == 'RBF':
            # Compute pairwise standardized Euclidean squared distances
            diff = X_norm[:, None, :] - X_norm[None, :, :]   # shape (N, N, D)
            dist2 = np.sum((diff**2) / W[None, None, :], axis=2)  # shape (N, N)

            # RBF covariance
            return sf2 * np.exp(-0.5 * dist2)
        else:
            print('ERROR no kernel with name', kernel)

    ################################
    # --- Covariance of sample --- #
    ################################               

    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        """
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        """
        # Broadcasted per-feature differences for all rows: (n_points, nx_dim)
        diff = Xnorm - xnorm[None, :]           # xnorm reshaped to (1, nx_dim) via [None, :]
        # Standardized Euclidean squared distance per row: (n_points,)
        dist2 = np.sum((diff ** 2) / ell[None, :], axis=1)
        # RBF covariance; reshape to (n_points, 1) to mirror cdist(..., ...)**2 output
        cov_matrix = (sf2 * np.exp(-0.5 * dist2)).reshape(-1, 1)
        return cov_matrix
    
    #############################
    # --- Covariance Matrix --- #
    #############################    
    # With cdist
    """
    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- decription ---
        '''
    
        if kernel == 'RBF':
            dist       = cdist(X_norm, X_norm, 'seuclidean', V=W)**2 
            cov_matrix = sf2*np.exp(-0.5*dist)
            return cov_matrix
            # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        else:
            print('ERROR no kernel with name ', kernel)

    ################################
    # --- Covariance of sample --- #
    ################################    
        
    def calc_cov_sample(self,xnorm,Xnorm,ell,sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''    
        # internal parameters
        nx_dim = self.nx_dim

        dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-.5*dist)

        return cov_matrix                
    """
    ###################################
    # --- negative log likelihood --- #
    ###################################   
    
    def negative_loglikelihood(self, hyper, X, Y):
        '''
        --- decription ---
        ''' 
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel          = self.kernel
        
        W               = np.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = np.exp(2*hyper[nx_dim])    # variance of the signal 
        sn2             = np.exp(2*hyper[nx_dim+1])  # variance of noise

        K       = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8)*np.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        L       = np.linalg.cholesky(K)            # do a cholesky decomposition
        logdetK = 2 * np.sum(np.log(np.diag(L)))   # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY   = np.linalg.solve(L,Y)             # obtain L^{-1}*Y
        alpha   = np.linalg.solve(L.T,invLY)       # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = np.dot(Y.T,alpha) + logdetK      # construct the NLL

        return NLL
    
    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################   
    
    def determine_hyperparameters(self):
        '''
        --- decription ---
        Notice we construct one GP for each output
        '''   
        # internal parameters
        X_norm, Y_norm  = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim  = self.kernel, self.ny_dim
        Cov_mat         = self.Cov_mat
        
        # In lb I had changed the lb right extreme to -10 and the ub right extreme to -1
        lb               = np.array([-4.]*(nx_dim+1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = np.array([4.]*(nx_dim+1) + [ -2.])  # ub on parameters (this is inside the exponential)
        bounds           = np.hstack((lb.reshape(nx_dim+2,1),
                                      ub.reshape(nx_dim+2,1)))
        multi_start      = self.multi_hyper                   # multistart on hyperparameter optimization
        multi_startvec   = sobol_seq.i4_sobol_generate(nx_dim + 2,multi_start)

        options  = {'disp':False,'maxiter':10000}          # solver options
        hypopt   = np.zeros((nx_dim+2, ny_dim))            # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.]*multi_start                        # values for multistart
        localval = np.zeros((multi_start))                 # variables for multistart

        invKopt = []
        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):    
            # --- multistart loop --- # 
            for j in range(multi_start):
                print('multi_start hyper parameter optimization iteration = ',j,'  input = ',i)
                hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood,hyp_init,args=(X_norm,Y_norm[:,i])\
                               ,method='SLSQP',options=options,bounds=bounds,tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
            minindex    = np.argmin(localval)
            hypopt[:,i] = localsol[minindex]
            ellopt      = np.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = np.exp(2.*hypopt[nx_dim,i])
            sn2opt      = np.exp(2.*hypopt[nx_dim+1,i]) + 1e-8

            # --- constructing optimal K --- #
            Kopt        = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt*np.eye(n_point)
            # --- inverting K --- #
            invKopt     += [np.linalg.solve(Kopt,np.eye(n_point))]

        return hypopt, invKopt

    ########################
    # --- GP inference --- #
    ########################     
    
    def GP_inference_np(self, x):
        '''
        --- decription ---
        '''
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample          = self.calc_cov_sample
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        var_out                  = self.var_out
        # Sigma_w                = self.Sigma_w (if input noise)

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(ny_dim)
        var   = np.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])

            # --- determine covariance of each output --- #
            k       = calc_cov_sample(xnorm,Xsample,ellopt,sf2opt)
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:,i])
            var[i]  = max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k)) # numerical error
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2
        
        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]

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
            # Ask acquisition function for next batch

            # Objective function batch evaluation
            random_selection = RandomSelection(self.X_searchspace, objective_func, self.batch)
            print(f"[Iter {iteration+1}/{self.iterations}] Best so far: {np.max(self.Y):.4f}")
            self.Y = np.concatenate([self.Y, random_selection.random_Y])
            self.time += [datetime.timestamp(datetime.now())-start_time]*(len(random_selection.random_Y))

'''
X_initial = ([[33, 6.25, 10, 20, 20, 'celltype_1'],
              [38, 8, 20, 10, 20, 'celltype_3'],
              [37, 6.8, 0, 50, 0, 'celltype_1'],
              [36, 6.0, 20, 20, 10, 'celltype_3'],
              [36, 6.1, 20, 20, 10, 'celltype_2'],
              [38, 6.0, 30, 50, 10, 'celltype_1']])

#temp = np.linspace(30, 40, 5)
#pH = np.linspace(6, 8, 5)
#f1 = np.linspace(0, 50, 5)
#f2 = np.linspace(0, 50, 5)
#f3 = np.linspace(0, 50, 5)
#celltype = ['celltype_1','celltype_2','celltype_3']

#X_searchspace     = [[a,b,c,d,e,f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]
'''
X_initial=()
X_searchspace = sobol_searchspace()
BO_m = BO(X_initial, X_searchspace, 15, 5, objective_func)