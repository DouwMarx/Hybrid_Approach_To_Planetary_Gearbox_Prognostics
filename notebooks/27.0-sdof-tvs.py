import numpy as np
import matplotlib.pyplot as plt
import src.models.lumped_mas_model as pglmm
import src.models.diagnostics as diag

m = 10
c = 5
F = 4

k_mean = 1000
delta_k = 450

X0 = np.array([F / (k_mean + delta_k), 0])
# X0 = np.array([0, 0])
# X0 = np.array([F / (k_mean), 0])


t_range = np.linspace(0, 10, 1000)
t_step_start = 0
t_step_duration = 1

sdof_dict = {"m": m,
             "c": c,
             "F": F,
             "delta_k": delta_k,
             "k_mean": k_mean,
             "X0": X0,
             "t_range": t_range,
             "t_step_start": t_step_start,
             "t_step_duration": t_step_duration,
             "tvms_type": "sine_mean_delta_drop"}

sho1 = pglmm.SimpleHarmonicOscillator(sdof_dict)

sol,t  = sho1.get_transducer_vibration()
sho1.plot_sol()

# optfor = {"m": [9, 11],
#           #"c": [4, 6],
#          # "F": [3, 5],
#          # "delta_k": [400, 500],
#           "k_mean": [8000, 1200]}
#
# d = diag.Diagnostics(sol, optfor, sdof_dict, pglmm.SimpleHarmonicOscillator)
#
# d.do_optimisation()
