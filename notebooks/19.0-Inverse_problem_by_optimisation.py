
import numpy as np
import matplotlib.pyplot as plt
import src.features.second_order_solvers as s
import scipy.optimize as opt
import time
import src.features.parameter_inference as pi

plt.close("all")


# True system
############################################################################333
Km = np.array([[400,-200,0],
              [-200, 400,-200],
              [0, -200, 200]])

K = lambda time: Km

C = np.array([[0.55,-0.2,0],
              [-0.2, 0.55,-0.2],
              [0, -0.2, 0.35]])

M = np.eye(3)
M[0, 0] = 1.5

f = np.zeros((3, 1))
X0 = np.zeros((3, 1))
Xd0 = np.ones((3, 1))*0.8

t = np.linspace(0, 10, 1000)

# Generate actual and measured response
lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
true = lmm.solve_de("RK")

measured = np.random.normal(true, 0.08)




#Definition of objective functions
def cost_m1_only(theta):
    M = np.eye(3)
    M[0,0] = theta[0]
    lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
    sol = lmm.solve_de("RK")

    return np.linalg.norm(sol - measured)

def cost_m1_init_cond(theta):
    M = np.eye(3)
    M[0, 0] = theta[0]
    X0 = np.array([theta[1:4]]).T
    Xd0 = np.array([theta[4:]]).T

    lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
    sol = lmm.solve_de("RK")
    return np.log(np.linalg.norm(sol - measured))

def cost_m1_init_cond_meas1_state(theta):
    M = np.eye(3)
    M[0, 0] = theta[0]
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

# Find system mass given all of the system response as measurement
#t_start = time.time()
#theta0 = np.array([2])
#optimum_m1 = opt.minimize(cost_m1_only, theta0)
#print("Find m1")
#print(optimum_m1)
#print("Done in", time.time()-t_start, " seconds")


# Find the system mass and intial conditions given all states as measured
#t_start = time.time()
#bnds = [(0.5, 2),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1)]

#startpoint = np.array([1, 0, 0, 0, 0, 0, 0])

#optimum_m1_init = opt.differential_evolution(cost_m1_init_cond, bnds, polish=True, maxiter=10)
#optimum_m1_init = opt.minimize(cost_m1_init_cond_meas1_state, x0=startpoint, bounds=bnds, method='L-BFGS-B')
#print("Find m1 and initial conditions")
#print(optimum_m1_init)
#print("Done in", time.time()-t_start, " seconds")


# Find mass and intial conditions given only one state in measureable
t_start = time.time()
#bnds = [(0.5, 5),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1),
#        (-1, 1)]

bnds = np.array([[0.5, 4],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1]])

startpoint = np.array([1, 0, 0, 0, 0, 0, 0])

#optimum_m1_init_1state = opt.minimize(cost_m1_init_cond_meas1_state,
#                                      x0=startpoint,
#                                      bounds=bnds,
#                                      method='L-BFGS-B',
#                                      options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-12, 'gtol': 1e-12, 'eps': 1e-12, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
optimum_m1_init_1state = opt.differential_evolution(cost_m1_init_cond_meas1_state, bnds, polish=True)#, popsize=10, maxiter=0)#, , popsize=100)
#optimum_m1_init_1state = opt.minimize(cost_m1_init_cond_meas1_state, x0=startpoint, bounds=bnds, method='L-BFGS-B', options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-12, 'gtol': 1e-12, 'eps': 1e-12, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
#optimum_m1_init_1state = opt.basinhopping(cost_m1_init_cond_meas1_state, x0=startpoint)#, maxiter=0, popsize=100)
print("Find m1 and initial conditions, measure 1 state")
#print(optimum_m1_init_1state)
#print("Done in", time.time()-t_start, " seconds")

#best_sol, sols = pi.random_init(cost_m1_init_cond_meas1_state, startpoint, bnds, 100)

print("Best solution")
print(best_sol)



# Plot the measured and actual response as well as the approximation by optimisation
plt.figure("Measured and true response of one state (Acceleration)")
plt.plot(t, true[:, -3])
plt.scatter(t, measured[:, -3])
plt.xlabel("time [s]")
plt.ylabel("Response")


solved = optiminum_plot(best_sol['x'])
plt.plot(t, solved[:, -3])