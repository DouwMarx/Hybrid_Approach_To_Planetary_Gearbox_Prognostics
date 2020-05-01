import numpy as np
import scipy.integrate as inter
import matplotlib.pyplot as plt

class RungeKutta(object):
    def __init__(self, M, C, K, f ,X0 ,Xd0, time_range, solver_options={"beta_1": 0.5, "beta_2": 0.25}):
        self.M = M
        self.C = C
        self.K = K
        self.X0 = X0
        self.Xd0 = Xd0
        self.f = f
        self.solver_options = solver_options

        self.time_range = time_range

        self.dt = np.average(np.diff(self.time_range))
        self.dimensionality = np.shape(M)[0]

        self.dim = np.shape(self.M)[0]  # Matrix dimensionality

        self.beta_1 = self.solver_options["beta_1"]
        self.beta_2 = self.solver_options["beta_2"]

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

class Newmark_Beta(object):
    """
    beta1 = gamma
    beta2 = beta
    """
    def constants(self):
        """
        Prevents the re-computation of constant values
        Returns
        -------

        """
        T1 = 2/(self.beta_2*self.dt**2)
        T2 = 2*self.beta_1/(self.beta_2*self.dt)
        T3 = 2/(self.beta_2*self.dt)
        T4 = (1-self.beta_2)/self.beta_2
        T5 = 1 - 2*self.beta_1/self.beta_2
        T6 = (1 - self.beta_1/self.beta_2)*self.dt

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6


        sqaure = np.ones((self.dimensionality,self.dimensionality))
        top = np.hstack((-T1*sqaure, -T3*sqaure, -T4*sqaure))
        bot = np.hstack((-T2*sqaure, +T5*sqaure, +T6*sqaure))
        self.Uh_np_mat = np.vstack((top, bot))

        return

    def get_Xdd0(self):
        """
        Computes the initial accelerations given the initial conditions
        Returns
        -------
        """

        T1 = -np.dot(np.linalg.solve(self.M, self.K(self.time_range[0])), self.X0)
        T2 = -np.dot(np.linalg.solve(self.M, self.C), self.Xd0)
        T3 = np.linalg.solve(self.M, self.f)

        return T1 + T2 + T3

    def Xp(self, Xdhp, Xddhp,t):
        b = self.f + np.dot(self.C, Xdhp) + np.dot(self.M, Xddhp)
        return - np.linalg.solve(self.A(t), b)

    def Xddhp(self, X, Xd, Xdd):
        return -self.T1*X - self.T3*Xd - self.T4*Xdd

    def Xdhp(self, X, Xd, Xdd):
        return -self.T2*X + self.T5*Xd + self.T6*Xdd  # dt allready acounted for in constants function


    def Xddp(self, Xddhp, Xp):
        return Xddhp + self.T1*Xp

    def Xdp(self, Xdhp, Xp):
        return Xdhp + self.T2*Xp

    def U_np(self, U_n, u_np):
        return self.Uh_np(U_n) + np.vstack((u_np*self.T1, u_np*self.T2))

    def Uh_np(self, U_n):
        return np.dot(self.Uh_np_mat, U_n)

    def A(self, t):
        return self.T1*self.M + self.T2*self.C + self.K(t)

    def u_p(self, U_n, t):
        fCuMu = self.f + np.dot(np.hstack((self.M, self.C)), self.Uh_np(U_n))
        return - np.linalg.solve(self.A(t), fCuMu)

    def newmark_solve(self):
        self.constants()

        sol_len = len(self.time_range)
        sol = np.zeros((self.dimensionality*3, sol_len))

        Xdd = self.get_Xdd0()
        Xd = self.Xd0
        X = self.X0

        sol[:, 0] = np.vstack((X, Xd, Xdd))[:, 0]

        for time_step, time_index in zip(self.time_range, range(1, sol_len)):
            Xddhp = self.Xddhp(X, Xd, Xdd)
            Xdhp = self.Xdhp(X, Xd, Xdd)

            Xp = self.Xp(Xdhp, Xddhp, time_step)
            Xddp = self.Xddp(Xddhp, Xp)
            Xdp = self.Xdp(Xdhp, Xp)

            X = Xp
            Xd = Xdp
            Xdd = Xddp

            sol[:, time_index] = np.vstack((X, Xd, Xdd))[:, 0]
        return sol.T

    def solve(self):

        UUn = self.U0
        sol = np.zeros((np.shape(UUn)[0], np.shape(self.time)[0]))
        for i, t in zip(range(len(self.time)), self.time):

            u_np = self.u_p(UUn, t)
            U_p = self.U_np(UUn, u_np)

            UUn = np.vstack((u_np, U_p[self.dimensionality:, :], U_p[0:self.dimensionality, :]))

            sol[:, i] = UUn[:, 0]

        return sol

class LMM_sys(RungeKutta, Newmark_Beta):
    """
    Lumped mass model object for second order equation of Newtons second law
    Time variable stiffness, All other parameters are constant
    """

    def solve_de(self, solver):
        if solver == "RK":
            return self.rk_solve()

        if solver == "Newmark":
            return self.newmark_solve()

        else:
            print("not implemented")

