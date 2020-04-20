import src.features.newmark as n
import models.lumped_mas_model.llm_models as lmm
import src.features.newmark as n
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

#PG = lmm.make_chaari_2006_model()
PG = lmm.make_chaari_2006_1planet()

Km = np.array([[400,-200,0],
              [-200, 400,-200],
              [0, -200, 200]])

K = lambda time: Km


C = np.array([[0.55,-0.2,0],
              [-0.2, 0.55,-0.2],
              [0, -0.2, 0.35]])

M = np.eye(3)

f = np.zeros((3, 1))

X0 = np.zeros((3, 1))

Xd0 = np.ones((3, 1))

betas = {"beta_1": 0.5,  # lambda 0.5
         "beta_2": 1/4}     # beta 0.25

matrices = {"M": M,
            "K": K,
            "C": C,
            "f": f}

initial_conditions = {"X0": X0,
                      "Xd0": Xd0}

t = np.linspace(0, 10, 10000)

problem = n.Newmark_int(betas, matrices, initial_conditions, t)

plt.figure()
s = problem.solve()
#plt.plot(t, s.T[:,0])#[:, 0:M.shape[0]])
plt.plot(t, s.T)#[:, 0:M.shape[0]])
plt.xlabel("time [s]")
plt.ylabel("Response")
#plt.ylim(-1E4, 1E4)