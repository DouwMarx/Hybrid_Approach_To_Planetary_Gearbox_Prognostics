import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models
import importlib  #This allows me to reload my own module every time

import src.features.proc_lib as proc
importlib.reload(pglmm)


plt.close("all")

PG = lmm_models.make_chaari_2006_1planet()
#PG = lmm_models.make_lin_1999_model()

t = np.linspace(0,1,1000000)

d1 = pglmm.DE_Integration(PG)
X0 = d1.X_0()
X0 = np.array([[-1.67427896e-15,  1.81686928e-07,  1.81686884e-08,  3.54914606e-08,
       -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
        9.88800572e-07, -3.34859651e-15,  3.81542498e-07, -5.11431563e-07,
        4.10034386e-14,  9.21551453e-09,  9.21668277e-10,  1.80039978e-09,
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
    sol = d1.Run_Integration(X0, t)

    half_sol_shape = int(0.5*np.shape(sol)[1])

    #

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
    plt.figure("Displacements")
    plt.ylabel("Displacement [m]")
    plt.xlabel("Time [s]")
    #p = plt.plot(t, sol[:,0:half_sol_shape])
    #plt.ylim([-1e-6,1e-6])
    #plt.legend(iter(p),lables)
    plt.plot(t, sol[:, [3,4]])

    plt.figure("Velocities")
    #p = plt.plot(t, sol[:,half_sol_shape:])
    plt.ylabel("Velocity [m/s]")
    plt.xlabel("Time [s]")
    #plt.legend(iter(p),lables)
    #plt.ylim([-1e-9,1e-9])
    plt.plot(t, sol[:, [3 + half_sol_shape, 4 + half_sol_shape]])
    plt.show()

    plt.figure("Ring Planet Stiffness")
    plt.plot(t, PG.k_rp(t))
    plt.xlabel("Time [s]")
    plt.ylabel("Mesh Stiffness [N/m")

    plt.figure("Sun Planet Stiffness")
    plt.plot(t, PG.k_sp(t))
    plt.xlabel("Time [s]")
    plt.ylabel("Mesh Stiffness [N/m")

    d = sol[:,3]
    freq, mag, phase = fft(d,np.average(np.diff(t)))
    plt.figure()
    plt.plot(freq, mag)

    return sol

s = run_sol()