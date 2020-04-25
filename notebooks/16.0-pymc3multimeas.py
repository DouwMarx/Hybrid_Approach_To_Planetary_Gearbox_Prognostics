import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

if __name__ == '__main__':
    timerange = np.linspace(1, 10, 15)

    alpha = 2
    beta = 2
    #  generate some data

    y = np.zeros((len(timerange)))
    for i in range(len(timerange)):
        y[i] = alpha + beta*timerange[i]**2 + np.random.rand()

    y = [y, y + np.random.rand(len(y)) ]


    def sys_model(alpha, beta):

        return alpha + beta * timerange**2

    #plt.figure()
    #plt.scatter(timerange, y)

    time_varying_model = pm.Model()

    with time_varying_model:

        #  Set up priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta",mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        #  System model
        mu = sys_model(alpha, beta)

        #  Likelihood of observations
        Y = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Sampler to use
        step = pm.NUTS()
        trace = pm.sample(5000, chains=2, cores=1, step=step,tune=1000)

    #map_estimate = pm.find_MAP(model=sys_model)
    #print("Map_estimate", map_estimate)

    # summary of stats
    print(pm.summary(trace))
    pm.traceplot(trace)

    # plot the result
    #plt.plot(timerange, sys_model(map_estimate['alpha'], map_estimate['beta']))

