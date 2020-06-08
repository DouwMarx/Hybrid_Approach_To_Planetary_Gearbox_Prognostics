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
t_step_start = 1
t_step_duration = 1

sdof_dict = {"m": m,
             "c": c,
             "F": F,
             "delta_k": delta_k,
             "k_mean": k_mean,
             "X0": X0,
             "t_range": t_range,
             "t_step_start": t_step_start,
             "t_step_duration": t_step_duration}

sho1 = pglmm.SimpleHarmonicOscillator(sdof_dict, "dropstep")

sol = sho1.integrate_ode()
# sho1.plot_sol()

optfor = {"m": [0, 0],
          "c": [0, 0],
          "F": [0, 0],
          "delta_k": [0, 0],
          "k_mean": [0, 0],
          "X0": [0, 0]}

d = diag.Diagnostics(sol["y"], optfor, sdof_dict)
