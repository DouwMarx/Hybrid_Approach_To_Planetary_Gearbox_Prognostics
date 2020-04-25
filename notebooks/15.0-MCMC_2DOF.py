import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import src.features.second_order_solvers as s
import theano.tensor as tt
from theano.compile.ops import as_op

if __name__ == '__main__':

    @as_op(itypes=[tt.dscalar], otypes=[tt.dvector])
    def mu(m1):
        M = np.eye(dim)
        M[0, 0] = m1
        lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
        sol = lmm.solve_de("RK")
        return sol[:, -1]

    def make_measurements(m1):
        M = np.eye(dim)
        M[0, 0] = m1
        lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
        sol = lmm.solve_de("RK")
        return sol[:, -1]

    def sys_model_deterministic(m1):
        M = np.eye(dim)
        M[0, 0] = m1
        lmm = s.LMM_sys(M, C, K, f, X0, Xd0, t)
        sol = lmm.solve_de("RK")
        return sol[:, -1]

    plt.close("all")


    dim = 2
    Km = np.array([[400, -100],
                  [-100, 200]])
    K = lambda time: Km
    C = np.array([[0.55, -0.2],
                  [-0.2, 0.35]])

    f = np.zeros((dim, 1))
    X0 = np.zeros((dim, 1))
    Xd0 = np.ones((dim, 1))
    t = np.linspace(0, 0.1, 10)

    #  Make data (Observations) and plot them (Could be noisey, if so, we assume this noise is gaussian)
    Y = make_measurements(1) #+ np.random.rand(len(t)) # Answer we are looking for is m1 = 1kg

    #plt.figure("Measurements")
    #plt.scatter(t, Y)
    #plt.xlabel("time [s]")
    #plt.ylabel("Response")


    #if __name__ == '__main__':


    lmm_model = pm.Model()

    with lmm_model:

        #  Set up priors
        BoundedNormal = pm.Bound(pm.Normal, lower=0.1)
        m_1 = BoundedNormal("m1", mu=1, sigma=0.1)

        sigma = pm.Normal("sigma", mu =0, sigma=0.001)

        #  System model (Expected value of outcome)
        mu_expected = mu(m_1)

        #  Likelihood of observations under the assumption of Gaussian measurement noise
        Y_obs = pm.Normal("Y_obs", mu=mu_expected, sigma=sigma, observed=Y)

        # Sampler to use
        #step = pm.NUTS()
        trace = pm.sample(5000, cores=1, init=None)#, tune=10)#, chains=8, cores=4, step=step)

    #map_estimate = pm.find_MAP(model = lmm_model)
    #print("Map_estimate", map_estimate)

    # summary of stats
    #print(pm.summary(trace))

    # plot the result
    #plt.plot(t, sys_model_deterministic(map_estimate["m1"]))
