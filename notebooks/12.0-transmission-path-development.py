
import models.lumped_mas_model.llm_models as lmm
import numpy as np
import matplotlib.pyplot as plt
import definitions
import src.models.lumped_mas_model as pglmm

plt.close("all")

#PG = lmm.make_chaari_2006_model()
PG = lmm.make_chaari_2006_1planet()

# Calculate accelerations for the calculated velocities and displacements
d = definitions.root + "\\data\\external\\lmm\\Response_3_0.npy"
sol = np.load(d)
t = np.linspace(0, 1, 100000)


transp = pglmm.Transmission_Path(PG, sol)

plt.figure()
#plt.plot(transp.d_ri(1))
plt.plot(transp.F_ri(1, t))


