import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm


alpha = 2
beta = 2
#  generate some data
t = np.linspace(0,1,10)
y = alpha + beta*t + np.random.rand(len(t))

plt.figure()
plt.scatter(t, y)

line_model = pm.Model()

with line_model:

    #  Set up priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta",mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    #  System model
    mu = alpha + beta*t

    #  Likelihood of observations
    Y = pm.Normal("y", mu=mu, sigma=sigma, observed=y)


map_estimate = pm.find_MAP(model = line_model)