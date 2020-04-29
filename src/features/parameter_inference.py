import scipy.optimize as opt
import numpy as np
from tqdm import tqdm

def random_init(function,x0,bounds,n_runs):
    startpoints = np.random.rand(n_runs,len(x0))
    startpoints = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * startpoints

    solutions = []
    best_sol = 0
    lowest_cost = 999999
    for startpoint in tqdm(startpoints):
        sol = opt.minimize(function,
                          x0=startpoint,
                          bounds=bounds,
                          method='L-BFGS-B',
                          options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-12,
                                   'gtol': 1e-12, 'eps': 1e-12, 'maxfun': 15000, 'maxiter': 15000,
                                   'iprint': -1, 'maxls': 20})

        if sol["fun"] < lowest_cost:
            best_sol = sol
            lowest_cost = sol["fun"]

        solutions.append(sol)
    return best_sol, solutions



