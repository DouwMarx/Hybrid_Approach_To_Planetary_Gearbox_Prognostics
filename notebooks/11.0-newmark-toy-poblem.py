
import numpy as np
import matplotlib.pyplot as plt
import src.features.second_order_solvers as s
import time
import scipy.signal as sig

plt.close("all")

Km = np.array([[400,-200,0],
              [-200, 400,-200],
              [0, -200, 200]])

def K(t):
    freq = 1
    p = 200
    return Km + np.array([[1, -1, 0], [-1, 0, 0],[0, 0, 1]])*p*0.5*(1 + sig.square(t*2*np.pi*freq))#np.sin(t*2*np.pi*freq)


C = np.array([[0.55,-0.2,0],
              [-0.2, 0.55,-0.2],
              [0, -0.2, 0.35]])


f = np.zeros((3, 1))
f[0, 0] = 100

X0 = np.zeros((3, 1))
Xd0 = np.ones((3, 1))

M = np.eye(3)

t = np.linspace(0, 10, 10000)

lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)

tn = time.time()
sol_newmark = lmm.solve_de("Newmark")
tn = time.time() - tn
print("Newmark in ", tn, "seconds")

tr = time.time()
sol_runge = lmm.solve_de("RK")
tr = time.time() - tr
print("Runge Kutta in ", tr, "seconds")





plt.figure()
plt.plot(t, sol_runge[:, -3], label = "Runge Kutta")
plt.plot(t, sol_newmark[:, -3], label = "Newmark")
plt.legend()
plt.xlabel("time [s]")
plt.grid()
plt.ylabel("Acc")
