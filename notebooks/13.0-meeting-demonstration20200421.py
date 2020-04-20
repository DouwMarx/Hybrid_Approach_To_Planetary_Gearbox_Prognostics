
import models.lumped_mas_model.llm_models as lmm
import numpy as np
import matplotlib.pyplot as plt
import definitions
import src.models.lumped_mas_model as pglmm

plt.close("all")

#PG = lmm.make_chaari_2006_model()
PG = lmm.make_chaari_2006_1planet()

# Calculate accelerations for the calculated velocities and displacements
#d = definitions.root + "\\data\\external\\lmm\\Response_3_0.npy"
#sol = np.load(d)
t = np.linspace(0, 1, 100000)
#d1 = pglmm.DE_Integration(PG)
#xdd = d1.X_dotdot(sol, t)
d = definitions.root + "\\" + "data\\external\\lmm\\"
#np.save(d + "Acceleration_3_0.npy", xdd)


xdd = np.load(d + "Acceleration_3_0.npy")
plt.figure("Accelerations")
plt.ylabel("Acceleration [m]")
plt.xlabel("Time [s]")
#p = plt.plot(t, sol[:,0:half_sol_shape])
#plt.ylim([-1e-6,1e-6])
#plt.legend(iter(p),lables)
plt.plot(t, xdd[:, 4])






