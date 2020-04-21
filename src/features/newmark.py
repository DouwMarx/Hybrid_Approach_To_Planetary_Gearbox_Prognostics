import numpy as np

class Newmark_int(object):
    """
    A class for solving second order differential equations of mechanical systems

    Note that proportional damping is not time varying like stiffness
    """

    def __init__(self, betas, matrices, initial_conditions,time_range):
        """
        Intializes the new mark algorithm
        Parameters
        ----------
        betas: dict {"beta_1":Float, "beta2":float}
        matrices: dict {"M": Array, "K": Array, "C": Array, "f"}
        initial_conditions: dict {"X0": Array, "Xd0":Array}
        """

        self.beta_1 = betas["beta_1"]
        self.beta_2 = betas["beta_2"]

        self.M = matrices["M"]
        self.K = matrices["K"]
        self.C = matrices["C"]
        self.f = matrices["f"]

        self.u0 = initial_conditions["X0"]
        self.ud0 = initial_conditions["Xd0"]

        self.time = time_range
        self.dt = np.average(np.diff(time_range))

        self.dimensionality = np.shape(self.u0)[0]

        self.constants() #Initializes constants

        self.udd0 = self.get_udd0()  # Compute initial accelerations

        self.U0 = np.vstack((self.u0, self.ud0, self.udd0))
        return

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


        sqaure = np.ones((self.dimensionality,self.dimensionality))
        top = np.hstack((-T1*sqaure, -T3*sqaure, -T4*sqaure))
        bot = np.hstack((-T2*sqaure, +T5*sqaure, +T6*sqaure))
        self.Uh_np_mat = np.vstack((top, bot))

        return

    def get_udd0(self):
        """
        Computes the initial accelerations given the initial conditions
        Returns
        -------
        """

        T1 = -np.dot(np.linalg.solve(self.M, self.K(self.time[0])), self.u0)
        T2 = -np.dot(np.linalg.solve(self.M, self.C), self.ud0)
        T3 = np.linalg.solve(self.M, self.f)

        return T1 + T2 + T3

    def U_np(self, U_n, u_np):
        return self.Uh_np(U_n) + np.vstack((u_np*self.T1, u_np*self.T2))

    def Uh_np(self, U_n):
        return np.dot(self.Uh_np_mat, U_n)

    def A(self, t):
        return self.T1*self.M + self.T2*self.C + self.K(t)

    def u_p(self, U_n, t):
        fCuMu = self.f + np.dot(np.hstack((self.M, self.C)), self.Uh_np(U_n))
        return - np.linalg.solve(self.A(t), fCuMu)

    def solve(self):

        UUn = self.U0
        sol = np.zeros((np.shape(UUn)[0], np.shape(self.time)[0]))
        for i, t in zip(range(len(self.time)), self.time):

            u_np = self.u_p(UUn, t)
            U_p = self.U_np(UUn, u_np)

            UUn = np.vstack((u_np, U_p[self.dimensionality:, :], U_p[0:self.dimensionality, :]))

            sol[:, i] = UUn[:, 0]

        return sol