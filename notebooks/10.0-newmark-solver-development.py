import src.features.newmark as n
import models.lumped_mas_model.llm_models as lmm
import src.features.newmark as n
import numpy as np
import matplotlib.pyplot as plt
import importlib
#importlib(newmark)
plt.close("all")

#PG = lmm.make_chaari_2006_model()
PG = lmm.make_chaari_2006_1planet()

K = PG.K
C = PG.C
M = PG.M
f = PG.T

X0 = np.zeros((M.shape[0], 1))
#X0[0, 0] = 1E-9

Xd0 = np.zeros((M.shape[0], 1))

betas = {"beta_1": 0.5,  # lambda 0.5
         "beta_2": 1/6}     # beta 0.25

matrices = {"M": M,
            "K": K,
            "C": C,
            "f": f}

initial_conditions = {"X0": X0,
                      "Xd0": Xd0}

t = np.linspace(0, 0.1, 10000)

problem = n.Newmark_int(betas, matrices, initial_conditions, t)

plt.figure()
s = problem.solve()
plt.plot(t, s.T[:,0])#[:, 0:M.shape[0]])
plt.xlabel("time [s]")
plt.ylabel("Response")
plt.ylim(-1E4, 1E4)