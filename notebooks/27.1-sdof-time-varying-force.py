import numpy as np
import matplotlib.pyplot as plt
import src.models.lumped_mas_model as pglmm
import src.models.diagnostics as diag

m = 1
c = 5
F = 60
k = 10000

# X00 = np.array([F / k, 0])

#X00 = np.array([0, 0])

t_range = np.linspace(0, 5, 10000)
t_step_start = 3
t_step_duration = 3

freq = 20
t_step_start = 0.1
t_step_duration = 3
f_low = 50
delta_f = -50


X00 = np.array([(delta_f+f_low) / k, 0])

def forcefunc(t):
    d = 20

    if t <= t_step_start:
        return f_low + delta_f

    elif t <= t_step_start + t_step_duration:
        return 0.5 * (np.cos(2 * np.pi * 0.5 * (
                t - t_step_start) / t_step_duration) + 1) * delta_f + f_low

    elif t <= t_step_start + t_step_duration + d:
        return f_low

    elif t <= t_step_start + t_step_duration + d + t_step_duration:
        return 0.5 * (-np.cos(
            np.pi * (
                    t - t_step_start - t_step_duration - d) / t_step_duration) + 1) * delta_f + f_low

    else:
        return delta_f + f_low


sdof_dict = {"m": m,
             "c": c,
             "f_func": forcefunc,
             "k": k,
             "X00": X00,
             "t_range": t_range,
             "tvms_type": "sine_mean_delta_step"}

sho1 = pglmm.SHOConstantK(sdof_dict)

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
