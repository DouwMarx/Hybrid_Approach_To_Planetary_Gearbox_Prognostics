import numpy as np
import pickle
import definitions
import pystan


path = definitions.root + "\\models\\stan_runs\\acc_meas_high_stiff_tvms_improved_prior.stan_res"


with open(path, 'rb') as filename:
    run = pickle.load(filename)


