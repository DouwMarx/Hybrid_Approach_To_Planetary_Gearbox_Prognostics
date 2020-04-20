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
        T6 = 1 - self.beta_1/self.beta_2

        self.T1 = T1
        self.T2 = T2

        ones = np.ones((1, self.dimensionality))
        self.T1T2 = np.hstack((self.T1*ones, self.T2*ones))

        self.udd_vec = np.array([-T1, -T3, -T4, +T1])
        self.ud_vec = np.array([-T2, T5, T6, T2])

        sqaure = np.ones((self.dimensionality,self.dimensionality))
        top = np.hstack((-T1*sqaure, -T3*sqaure, -T4*sqaure))
        bot = np.hstack((-T2*sqaure, +T5*sqaure, +T6*sqaure))
        self.Uh_np_mat = np.vstack((top,bot))
        return



    def U_np(self,Uh_np ,u_np):
        return Uh_np + np.dot(self.T1T2, np.vstack((np.ones((self.dimensionality,1)), u_np)))

    def Uh_np(self, U_n):
        return np.dot(self.Uh_np_mat, U_n)

    def A(self,t):
        return self.T1*self.M + self.T2*self.C + self.K(t)

    def u_p(self, Uh_np, t):
        fCuMu = self.f + np.dot(np.hstack((self.M, self.C)), Uh_np)
        return -np.linalg.solve(self.A(t), fCuMu)