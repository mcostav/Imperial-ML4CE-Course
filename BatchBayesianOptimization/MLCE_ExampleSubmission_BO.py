import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import time
import sobol_seq
import scipy

def objective_func(X: list): 
    return(np.array(virtual_lab.conduct_experiment(X)))

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