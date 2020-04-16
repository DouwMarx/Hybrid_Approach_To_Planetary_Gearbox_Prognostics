import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models


plt.close("all")

PG = lmm_models.make_chaari_2006_model()
#PG = lmm_models.make_lin_1999_model()

t = np.linspace(0,0.00005,10000)

d1 = pglmm.DE_Integration(PG)
X0 = d1.X_0()


def run_sol():
    sol = d1.Run_Integration(X0, t)

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
    p = plt.plot(t, sol[:,0:3*3+4*3])
    #plt.ylim([-1e-6,1e-6])
    plt.legend(iter(p),lables)

    plt.figure("Velocities")
    p = plt.plot(t, sol[:,3*3+4*3:])
    plt.legend(iter(p),lables)
    #plt.ylim([-1e-9,1e-9])
    plt.show()
    return

run_sol()