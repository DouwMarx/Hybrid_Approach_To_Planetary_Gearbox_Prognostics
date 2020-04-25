
import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import theano
import time

if __name__ == '__main__':
    timestart = time.time()

    plt.style.use('seaborn-darkgrid')


    def SIR(y, t, p):
        ds = -p[0]*y[0]*y[1]
        di = p[0]*y[0]*y[1] - p[1]*y[1]
        return [ds, di]

    times = np.arange(0,5,0.25)

    beta,gamma = 4,1.0   #Actual model parameters
    # True system response
    y = odeint(SIR, t=times, y0=[0.99, 0.01], args=((beta,gamma),), rtol=1e-8)

    # Observed system response
    yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.2, 0.3])

    # Plot the actual and observed system response
    plt.plot(times[1::],yobs, marker='o', linestyle='none')
    plt.plot(times, y[:,0], color='C0', alpha=0.5, label=f'$S(t)$')
    plt.plot(times, y[:,1], color ='C1', alpha=0.5, label=f'$I(t)$')
    plt.legend()
    plt.show()

    #  Define the diferential equation for the pymc3odint module
    sir_model = DifferentialEquation(
        func=SIR,  # The DE function, first order
        times=np.arange(0.25, 5, 0.25),  # time range (why not use already defined time?
        n_states=2,  # Degrees of freedom
        n_theta=2,   # Model parameters
        t0=0,        # Start at t=0
    )

    with pm.Model():
        sigma = pm.HalfCauchy('sigma', 1, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occured
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2,3)
        lam = pm.Lognormal('lambda',pm.math.log(2),2)
        beta = pm.Deterministic('beta', lam*R0)

        sir_curves = sir_model(y0=[0.99, 0.01], theta=[beta, lam])

        Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=yobs)

        prior = pm.sample_prior_predictive()
        trace = pm.sample(2000,tune=1000, target_accept=0.9, cores=1)
        posterior_predictive = pm.sample_posterior_predictive(trace)

        data = az.from_pymc3(trace=trace, prior = prior, posterior_predictive = posterior_predictive)
        az.plot_posterior(data, round_to=2, credible_interval=0.95)


        print("hours taken: ", (time.time() - timestart)/3600)