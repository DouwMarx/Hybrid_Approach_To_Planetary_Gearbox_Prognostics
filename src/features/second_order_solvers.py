import numpy as np
import scipy.integrate as inter
import matplotlib.pyplot as plt






class RungeKutta(object):
    def __init__(self, M, C, K, f ,X0 ,Xd0, time_range):
        self.M = M
        self.C = C
        self.K = K
        self.X0 = X0
        self.Xd0 = Xd0
        self.f = f

        self.time_range = time_range

        self.dim = np.shape(self.M)[0]  # Matrix dimensionality

        return

    def E_Q(self, t):
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

        c_over_m = np.linalg.solve(self.M, self.C)
        k_over_m = np.linalg.solve(self.M, self.K(t))
        f_over_m = np.linalg.solve(self.M, self.f)


        half_dim = self.dim

        E = np.zeros((self.dim*2, self.dim*2))
        E[half_dim:, 0:half_dim] = -k_over_m
        E[half_dim:, half_dim:] = -c_over_m
        E[0:half_dim, half_dim:] = np.eye(half_dim)

        Q = np.zeros((2*self.dim, 1))
        Q[half_dim:, 0] = f_over_m[:, 0]

        return E, Q

    def X_dot(self, X, t):
        E, Q = self.E_Q(t)
        X_dot = np.dot(E, np.array([X]).T) + Q

        return (X_dot[:, 0])

    def rk_solve(self):
        sol = inter.odeint(self.X_dot, np.vstack((self.X0, self.Xd0))[:, 0], self.time_range)#,full_output=1)
        acc = self.get_Xdotdot(sol)
        return np.hstack((sol, acc))

    def get_Xdotdot(self, sol):
        """
        Calculates accelerations from computed displacements and velocities
        Parameters
        ----------
        sol

        Returns
        -------

        """

        #Minv = np.linalg.inv(self.M)

        #xdd = np.zeros((len(t), self.dim))
        #for timestep, i in zip(self.time_range, range(len(t))):
        #    acc = -np.dot(np.dot(Minv, self.K(timestep)), sol[i, 0:self.dim].T) - np.dot(np.dot(Minv, self.C), sol[i, self.dim:].T) + np.dot(Minv, self.f[:, 0])
        #    xdd[i, :] = acc[:]

        #return xdd
        XXd = np.zeros((len(self.time_range), 2*self.dim))
        for timestep, i in zip(self.time_range, range(len(self.time_range))):
            E, Q = self.E_Q(timestep)
            acc = np.dot(E, np.array([sol[i]]).T) + Q

            XXd[i, :] = acc[:, 0]

        return XXd[:, self.dim:]


class LMM_sys(RungeKutta):
    """
    Lumped mass model object for second order equation of Newtons second law
    Time variable stiffness, All other parameters are constant
    """

    def solve_de(self, solver):
        if solver == "RK":
            return self.rk_solve()

        else:
            print("not implemented")

