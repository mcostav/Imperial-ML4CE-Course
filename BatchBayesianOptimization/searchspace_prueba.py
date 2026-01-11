import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import time
import sobol_seq
import scipy
from scipy.optimize import minimize

temp = np.linspace(30, 40, 5)
pH = np.linspace(6, 8, 5)
f1 = np.linspace(0, 50, 5)
f2 = np.linspace(0, 50, 5)
f3 = np.linspace(0, 50, 5)
celltype = ['celltype_1','celltype_2','celltype_3']

X_searchspace     = [[a,b,c,d,e,f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]
print(len(X_searchspace))