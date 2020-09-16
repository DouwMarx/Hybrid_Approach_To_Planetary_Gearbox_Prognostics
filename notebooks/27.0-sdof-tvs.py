import numpy as np
import matplotlib.pyplot as plt
import src.models.lumped_mas_model as pglmm
import src.models.diagnostics as diag

m = 1
c = 5
F = 50

k_mean = 1000
delta_k = 450

X00 = np.array([F / (k_mean + delta_k), 0])
# X0 = np.array([0, 0])
# X0 = np.array([F / (k_mean), 0])


t_range = np.linspace(0, 20, 1000)
t_step_start = 3
t_step_duration = 3

sdof_dict = {"m": m,
             "c": c,
             "F": F,
             "delta_k": delta_k,
             "k_mean": k_mean,
             "X00": X00,
             "t_range": t_range,
             "t_step_start": t_step_start,
             "t_step_duration": t_step_duration,
             "tvms_type": "sine_mean_delta_step"}

sho1 = pglmm.SimpleHarmonicOscillator(sdof_dict)

sol, t = sho1.get_transducer_vibration()
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
