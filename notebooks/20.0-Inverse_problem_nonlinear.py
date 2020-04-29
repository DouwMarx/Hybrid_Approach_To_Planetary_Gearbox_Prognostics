
import numpy as np
import matplotlib.pyplot as plt
import src.features.second_order_solvers as s
import scipy.optimize as opt
import time
import scipy.signal as sig
import src.features.parameter_inference as pi

plt.close("all")


# True system
############################################################################333
Km = np.array([[400,-200,0],
              [-200, 400,-200],
              [0, -200, 200]])


def K(t):
    freq = 10
    p = 200
    return Km + np.array([[1, -1, 0], [-1, 0, 0],[0, 0, 1]])*p*0.5*(1 + sig.square(t*2*np.pi*freq))#np.sin(t*2*np.pi*freq)

C = np.array([[0.55,-0.2,0],
              [-0.2, 0.55,-0.2],
              [0, -0.2, 0.35]])

M = np.eye(3)
M[0, 0] = 1.5

f = np.zeros((3, 1))
f[0, 0] = 100

X0 = np.zeros((3, 1))
Xd0 = np.ones((3, 1))*0.8

t = np.linspace(0, 10, 10000)

# Generate actual and measured response
lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
true = lmm.solve_de("RK")

measured = np.random.normal(true, 0.08)



def cost_m1_init_cond_meas1_state(theta):


    def K(t):
        freq = 10
        return Km + np.array([[1, -1, 0], [-1, 0, 0], [0, 0, 1]]) * theta[0] * 0.5 * (1 + sig.square(t * 2 * np.pi * freq))

    X0 = np.array([theta[1:4]]).T
    Xd0 = np.array([theta[4:]]).T

    lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
    sol = lmm.solve_de("RK")
    return np.linalg.norm(sol[:, -3] - measured[:, -3])

def optiminum_plot(theta):
    M = np.eye(3)
    M[0, 0] = theta[0]
    X0 = np.array([theta[1:4]]).T
    Xd0 = np.array([theta[4:]]).T

    lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
    sol = lmm.solve_de("RK")
    return sol


# Find mass and intial conditions given only one state in measureable
t_start = time.time()


bnds = np.array([[100, 300],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1]])

startpoint = np.array([199, 0, 0, 0, 0, 0, 0])


#print("Find m1 and initial conditions, measure 1 state")
#best_sol, sols = pi.random_init(cost_m1_init_cond_meas1_state, startpoint, bnds, 100)
optimum_m1_init_1state = opt.differential_evolution(cost_m1_init_cond_meas1_state, bnds, polish=True)
#print("Best solution")
#print(best_sol)



# Plot the measured and actual response as well as the approximation by optimisation
plt.figure("Measured and true response of one state (Acceleration)")
plt.plot(t, true[:, -3])
plt.scatter(t, measured[:, -3])
plt.xlabel("time [s]")
plt.ylabel("Response")


solved = optiminum_plot(best_sol['x'])
plt.plot(t, solved[:, -3])