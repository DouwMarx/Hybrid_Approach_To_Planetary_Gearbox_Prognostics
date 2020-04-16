import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models
import importlib  #This allows me to reload my own module every time
importlib.reload(pglmm)


plt.close("all")

PG = lmm_models.make_chaari_2006_1planet()
#PG = lmm_models.make_lin_1999_model()

t = np.linspace(0,0.1,100000)

d1 = pglmm.DE_Integration(PG)
X0 = d1.X_0()


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
    p = plt.plot(t, sol[:,0:half_sol_shape])
    #plt.ylim([-1e-6,1e-6])
    plt.legend(iter(p),lables)

    plt.figure("Velocities")
    p = plt.plot(t, sol[:,half_sol_shape:])
    plt.ylabel("Velocity [m/s]")
    plt.xlabel("Time [s]")
    plt.legend(iter(p),lables)
    #plt.ylim([-1e-9,1e-9])
    plt.show()
    return

run_sol()