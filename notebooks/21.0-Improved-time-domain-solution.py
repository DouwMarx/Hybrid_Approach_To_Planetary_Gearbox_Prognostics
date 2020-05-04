import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models
import importlib  #This allows me to reload my own module every time
import definitions
import src.features.second_order_solvers as s
import time



plt.close("all")

PG = lmm_models.make_chaari_2006_1planet()
#PG = lmm_models.make_lin_1999_model()

timerange = np.linspace(0, 0.1, 1000)


X0 = np.array([[-1.67427896e-15,  1.81686928e-07,  1.81686884e-08,  3.54914606e-08,
                -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
                9.88800572e-07, -3.34859651e-15,  3.81542498e-07, -5.11431563e-07]]).T
Xd0 = np.array([[4.10034386e-14,  9.21551453e-09,  9.21668277e-10,  1.80039978e-09,
                -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
                4.77182740e-08,  8.20079192e-14,  1.93539270e-08, -2.59430020e-08]]).T




def fft(data, fs):
    """

    Parameters
    ----------
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

def run_sol():
    solver = s.LMM_sys(PG.M, PG.C, PG.K, PG.T, X0, Xd0, timerange, solver_options={"beta_1": 0.4, "beta_2": 0.25})

    tnm = time.time()
    sol_nm = solver.solve_de("Newmark")
    print("Newmark time: ", time.time()-tnm)

    trk = time.time()
    sol_rk = solver.solve_de("RK")
    print("Runge Kutta time: ", time.time()-trk)

    half_sol_shape = int(0.5*np.shape(sol_nm)[1])

    lables = ("x_c",
              "y_c",
              "u_c",
              "x_r",
              "y_r",
              "u_r",
              "x_s",
              "y_s",
              "u_s",
              "zeta_1",
              "nu_1",
              "u_1",
              "zeta_2",
              "nu_2",
              "u_2",
              "zeta_3",
              "nu_3",
              "u_3")
    plt.figure("Acceleration")
    plt.ylabel("Displacement [m]")
    plt.xlabel("Time [s]")
    plt.plot(timerange, sol_rk[:, -3], label="Runge Kutta")
    plt.plot(timerange, sol_nm[:, -3], label="Newmark")
    plt.legend()
    plt.ylim(np.min(sol_rk[:, -3]),np.max(sol_rk[:, -3]))


    plt.figure("Ring Planet Stiffness")
    plt.plot(timerange, PG.k_rp(timerange))
    plt.xlabel("Time [s]")
    plt.ylabel("Mesh Stiffness [N/m")

    plt.figure("Sun Planet Stiffness")
    plt.plot(timerange, PG.k_sp(timerange))
    plt.xlabel("Time [s]")
    plt.ylabel("Mesh Stiffness [N/m")

    data = sol_nm[:, -3]
    freq, mag, phase = fft(data, 1/np.average(np.diff(timerange)))
    plt.figure()
    plt.plot(freq, mag)

    return sol_nm, sol_rk

s_nm, s_rk = run_sol()

d = definitions.root + "\\" + "data\\external\\lmm\\"
#np.save(d + "Response_4_0.npy", s)