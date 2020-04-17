import src.features.newmark as n
import models.lumped_mas_model.llm_models as lmm
import src.features.newmark as n
import numpy as np

PG = lmm.make_chaari_2006_model()

K = PG.K
C = PG.C
M = PG.M
f = PG.T
X0 = lmm.X0

betas = {"beta_1": 0.5,
         "beta_2": 0.25}

matrices = {"M": M,
            "K": K,
            "C": C,
            "f": f}

initial_conditions = {"X0": X0[0:int(X0.shape[0]/2)],
                      "Xd0": X0[int(X0.shape[0]/2):]}

t = np.linspace(0, 1, 10)

X = np.array([X0[0:int(X0.shape[0]/2)]]).T


U0 = np.vstack((X, X, X))

problem = n.Newmark_int(betas, matrices, initial_conditions, t)

