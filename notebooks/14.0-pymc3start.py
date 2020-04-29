import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

if __name__ == '__main__':
    alpha = 2
    beta = 2
    #  generate some data
    t = np.linspace(0, 1, 20)
    y = alpha + beta*t + np.random.rand(len(t))


    def sys_model(alpha, beta,t):
        return alpha + beta*t

    plt.figure()
    plt.scatter(t, y)

    line_model = pm.Model()

    with line_model:

        #  Set up priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta",mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        #  System model
        mu = sys_model(alpha,beta,t)

        #  Likelihood of observations
        Y = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Sampler to use
        step = pm.NUTS()
        trace = pm.sample(5000, chains=8, cores=4, step=step)

    map_estimate = pm.find_MAP(model = line_model)
    print("Map_estimate", map_estimate)

    # summary of stats
    print(pm.summary(trace))

    # plot the result
    plt.plot(t, sys_model(map_estimate['alpha'], map_estimate['beta'], t))

