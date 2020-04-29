
import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import theano
import time


timestart = time.time()
plt.close("all")



# Problem definition
dim = 2
Km = np.array([[400, -100],
               [-100, 200]])
K = lambda t: Km
C = np.array([[0.55, -0.2],
              [-0.2, 0.35]])

f = np.zeros((dim, 1))
x0 = np.zeros((dim, 1))
xd0 = np.ones((dim, 1))

X0 = np.vstack((x0, xd0))
timerange = np.linspace(0, 0.1, 10)

def Mass(p):
    m1 = p[0]
    return np.array([[m1, 0], [0, 1]])

def E_Q(t, p):
    """
    Converts the second order differential equation to first order (E matrix and Q vector)

    Parameters
    ----------
    t  : Float
         Time

    Returns
    -------
    E  : 2x(9+3xN) x 2x(9+3xN) Numpy array

    Based on Runge-kutta notes

    """

    M = Mass(p)
    c_over_m = np.linalg.solve(M, C)
    k_over_m = np.linalg.solve(M, K(t))
    f_over_m = np.linalg.solve(M, f)

    E = np.zeros((dim * 2, dim * 2))
    E[dim:, 0:dim] = -k_over_m
    E[dim:, dim:] = -c_over_m
    E[0:dim, dim:] = np.eye(dim)

    Q = np.zeros((2 * dim, 1))
    Q[dim:, 0] = f_over_m[:, 0]

    return E, Q


def X_dot(X, t, p):
    M = Mass(p)
    c_over_m = C#np.linalg.solve(M, C)
    k_over_m = K(t)#np.linalg.solve(M, K(t))
    f_over_m = f#np.linalg.solve(M, f)

    E = np.zeros((dim * 2, dim * 2))
    E[dim:, 0:dim] = -k_over_m
    E[dim:, dim:] = -c_over_m
    E[0:dim, dim:] = np.eye(dim)

    Q = np.zeros((2 * dim, 1))
    Q[dim:, 0] = f_over_m[:, 0]

    X_dot = np.dot(E, np.array([X]).T) + Q
    #X_dot = np.dot(E, X) + Q

    return X_dot[:,0]

    #return [p[0], X[1], X[2], 2*X[2]]


# True system response
X = odeint(X_dot, y0=X0[:, 0], t=timerange, args=tuple([[1]]))

# Observed system response
Xobs = np.random.normal(X,0.02)

# Plot the actual and observed system response
plt.plot(timerange, Xobs, marker='o', linestyle='none')
plt.plot(timerange, X, color='C0', alpha=0.5)

plt.legend()
plt.show()



#  Define the diferential equation for the pymc3odint module
lmm_model = DifferentialEquation(
    func=X_dot,  # The DE function, first order
    times=np.linspace(0, 0.1, 10),  # time range (why not use already defined time?
    n_states=4,  # Degrees of freedom
    n_theta=1,   # Model parameters
    t0=0,        # Start at t=0
)

with pm.Model():
    sigma = pm.HalfNormal('sigma', 0.05, shape=4)

    #m1 is bounded because mass cannot take negative value
    m1 = pm.Bound(pm.Normal, lower=0)('m1', 0.5, 3)

    # Set intitial conditions to known values, only parameter is mass
    lmm_sol = lmm_model(y0=X0, theta=[m1])

    Y = pm.Lognormal('Y', mu=pm.math.log(lmm_sol), sd=sigma, observed=Xobs)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(200, tune=100, target_accept=0.9, trace=1, cores=1)
    posterior_predictive = pm.sample_posterior_predictive(trace)

    data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive = posterior_predictive)
    az.plot_posterior(data, round_to=2, credible_interval=0.95)


    print("hours taken: ", (time.time() - timestart)/3600)