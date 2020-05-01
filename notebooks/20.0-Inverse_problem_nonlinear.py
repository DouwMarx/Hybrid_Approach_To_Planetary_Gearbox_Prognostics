
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

measured = np.random.normal(true, 0.16)



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
    X0 = np.array([theta[1:4]]).T
    Xd0 = np.array([theta[4:]]).T

    f = np.zeros((3, 1))
    f[0, 0] = 100

    def K(t):
        freq = 10
        return Km + np.array([[1, -1, 0], [-1, 0, 0], [0, 0, 1]]) * theta[0] * 0.5 * (1 + sig.square(t * 2 * np.pi * freq))

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
optimum_m1_init_1state = opt.differential_evolution(cost_m1_init_cond_meas1_state, bnds, polish=True,disp=True)
#print("Best solution")
#print(best_sol)



# Plot the measured and actual response as well as the approximation by optimisation
plt.figure("Measured and true response of one state (Acceleration)")
plt.plot(t, true[:, -3])
plt.scatter(t, measured[:, -3])
plt.xlabel("time [s]")
plt.ylabel("Response")


solved = optiminum_plot(optimum_m1_init_1state['x'])
plt.plot(t, solved[:, -3])

#seconds, 10000 increments
"""
fun: 7.972368098771397
jac: array([4523.23565154, 7656.9481001, 5296.76943186, 6081.52746508,
            6232.87502188, 6039.05013463, 6685.63798678])
message: 'Optimization terminated successfully.'
nfev: 10649
nit: 96
success: True
x: array([1.99996881e+02, -2.47569690e-05, -5.26822797e-05, -1.85527696e-05,
          8.00634156e-01, 7.98542224e-01, 8.00497971e-01])
"""


# 10 seconds 10000 increments 0.16 noise
"""
fun: 15.953308585537986
jac: array([3699.41794816, 903.8761128, 2559.62153979, 2874.05354804,
            3631.88399852, 3472.92291654, 644.32653275])
message: 'Optimization terminated successfully.'
nfev: 9410
nit: 81
success: True
x: array([1.99972131e+02, -4.30809639e-04, -6.76035917e-04, -5.46227370e-04,
          8.01660155e-01, 7.98372241e-01, 7.96708351e-01])
"""