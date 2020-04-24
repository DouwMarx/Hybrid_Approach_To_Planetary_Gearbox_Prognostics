
import numpy as np
import matplotlib.pyplot as plt
import src.features.second_order_solvers as s

plt.close("all")


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


t = np.linspace(0, 10, 10000)

lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
sol = lmm.solve_de("RK")

plt.figure()
plt.plot(t, sol)
plt.xlabel("time [s]")
plt.ylabel("Response")

plt.figure()
plt.plot(t, sol[:,-3])
plt.xlabel("time [s]")
plt.grid()
plt.ylabel("Acc")