import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models
import importlib  # This allows me to reload my own module every time
import definitions
import src.features.second_order_solvers as s
import time
from importlib import reload
reload(s)


plt.close("all")

# PG = lmm_models.make_chaari_2006_1planet()
PG = lmm_models.make_chaari_2006_model()
# PG = lmm_models.make_lin_1999_model()

# timerange = np.linspace(0, 0.02, 100000)
timerange = np.linspace(0, 0.1, 10000)

# X0 = np.array([[-1.67427896e-15, 1.81686928e-07, 1.81686884e-08, 3.54914606e-08,
#                 -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
#                 9.88800572e-07, -3.34859651e-15, 3.81542498e-07, -5.11431563e-07]]).T
# Xd0 = np.array([[4.10034386e-14, 9.21551453e-09, 9.21668277e-10, 1.80039978e-09,
#                  -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
#                  4.77182740e-08, 8.20079192e-14, 1.93539270e-08, -2.59430020e-08]]).T

dim = 21
X0 = np.zeros((dim, 1))
Xd0 = np.zeros((dim, 1))


def fft(data, fs):
    """

    Parameters
    ----------
    fs
    data: String
        The heading name for the dataframe

    Returns
    -------
    freq: Frequency range
    magnitude:
    phase:
    """

    d = data
    length = len(d)
    Y = np.fft.fft(d) / length
    magnitude = np.abs(Y)[0:int(length / 2)]
    phase = np.angle(Y)[0:int(length / 2)]
    freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
    return freq, magnitude, phase


solver = s.LMM_sys(PG.M, PG.C, PG.K, PG.T, X0, Xd0, timerange, solver_options={"beta_1": 0.55, "beta_2": 0.4})

#tnm = time.time()
#sol_nm = solver.solve_de("Newmark")
#print("Newmark time: ", time.time() - tnm)

trk = time.time()
sol = solver.solve_de("RK")
print("Runge Kutta time: ", time.time() - trk)

# t_stiff = time.time()
# sol = solver.solve_de("Radau")
# print("Radau time: ", time.time() - t_stiff)

# t_stiff = time.time()
# sol_bdf = solver.solve_de("BDF")
# print("BDF: ", time.time() - t_stiff)

solver.plot_solution(sol, "Displacement")
#solver.plot_solution(sol_stiff, "Velocity")
#solver.plot_solution(sol_stiff, "Displacement")
#solver.plot_solution(sol_nm, "Acceleration")

#PG.plot_tvms(timerange)



#    return sol_nm, sol_rk


#plt.figure()
#plt.plot(s_rad["t"], s_rad["y"].T)

