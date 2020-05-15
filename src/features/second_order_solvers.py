import numpy as np
import scipy.integrate as inter
import matplotlib.pyplot as plt


class RungeKutta(object):

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

        c_over_m = np.linalg.solve(self.M, self.C(t))
        k_over_m = np.linalg.solve(self.M, self.K(t))
        f_over_m = np.linalg.solve(self.M, self.f(t))

        half_dim = self.dim

        E = np.zeros((self.dim * 2, self.dim * 2))
        E[half_dim:, 0:half_dim] = -k_over_m
        E[half_dim:, half_dim:] = -c_over_m
        E[0:half_dim, half_dim:] = np.eye(half_dim)

        Q = np.zeros((2 * self.dim, 1))
        Q[half_dim:, 0] = f_over_m[:, 0]

        return E, Q

    def X_dot(self, X, t):
        E, Q = self.E_Q(t)
        X_dot = np.dot(E, np.array([X]).T) + Q

        return (X_dot[:, 0])

    def rk_solve(self):
        sol = inter.odeint(self.X_dot, np.vstack((self.X0, self.Xd0))[:, 0], self.time_range)  # ,full_output=1)
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

        # Minv = np.linalg.inv(self.M)

        # xdd = np.zeros((len(t), self.dim))
        # for timestep, i in zip(self.time_range, range(len(t))):
        #    acc = -np.dot(np.dot(Minv, self.K(timestep)), sol[i, 0:self.dim].T) - np.dot(np.dot(Minv, self.C), sol[i, self.dim:].T) + np.dot(Minv, self.f[:, 0])
        #    xdd[i, :] = acc[:]

        # return xdd
        XXd = np.zeros((len(self.time_range), 2 * self.dim))
        for timestep, i in zip(self.time_range, range(len(self.time_range))):
            E, Q = self.E_Q(timestep)
            acc = np.dot(E, np.array([sol[i]]).T) + Q

            XXd[i, :] = acc[:, 0]

        return XXd[:, self.dim:]


class Stiff_DE(object):

    def E_Q_stiff(self, t):
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

        # c_over_m = np.linalg.solve(self.M, self.C(t))
        # k_over_m = np.linalg.solve(self.M, self.K(t))
        # f_over_m = np.linalg.solve(self.M, self.f(t))

        c_over_m = np.dot(self.M_inv, self.C(t))
        k_over_m = np.dot(self.M_inv, self.K(t))
        f_over_m = np.dot(self.M_inv, self.f(t))

        half_dim = self.dim

        E = np.zeros((self.dim * 2, self.dim * 2))
        E[half_dim:, 0:half_dim] = -k_over_m
        E[half_dim:, half_dim:] = -c_over_m
        E[0:half_dim, half_dim:] = np.eye(half_dim)

        Q = np.zeros((2 * self.dim, 1))
        Q[half_dim:, 0] = f_over_m[:, 0]

        return E, Q

    def X_dot_stiff(self, t, X):
        E, Q = self.E_Q_stiff(t)
        X_dot = np.dot(E, np.array([X]).T) + Q

        return (X_dot[:, 0])

    def stiff_solve(self, method):
        sol = inter.solve_ivp(self.X_dot_stiff,
                              [self.time_range[0], self.time_range[-1]],
                              np.vstack((self.X0, self.Xd0))[:, 0],
                              method=method,
                              dense_output=True,
                              t_eval=self.time_range,
                              rtol=1e-3,
                              atol=1e-6)

        y = sol.y.T

        return y
        # acc = self.get_Xdotdot(y)
        # return acc
        # return np.hstack((y, acc))

    def get_Xdotdot(self, sol):
        """
        Calculates accelerations from computed displacements and velocities
        Parameters
        ----------
        sol

        Returns
        -------

        """

        # Minv = np.linalg.inv(self.M)

        # xdd = np.zeros((len(t), self.dim))
        # for timestep, i in zip(self.time_range, range(len(t))):
        #    acc = -np.dot(np.dot(Minv, self.K(timestep)), sol[i, 0:self.dim].T) - np.dot(np.dot(Minv, self.C), sol[i, self.dim:].T) + np.dot(Minv, self.f[:, 0])
        #    xdd[i, :] = acc[:]

        # return xdd
        XXd = np.zeros((len(self.time_range), 2 * self.dim))
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

        # {"beta_1": 0.5, "beta_2": 0.25}
        self.beta_1 = 0.5
        self.beta_2 = 0.25

        T1 = 2 / (self.beta_2 * self.dt ** 2)
        T2 = 2 * self.beta_1 / (self.beta_2 * self.dt)
        T3 = 2 / (self.beta_2 * self.dt)
        T4 = (1 - self.beta_2) / self.beta_2
        T5 = 1 - 2 * self.beta_1 / self.beta_2
        T6 = (1 - self.beta_1 / self.beta_2) * self.dt

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6

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

    def Xp(self, Xdhp, Xddhp, t):
        b = -self.f + np.dot(self.C, Xdhp) + np.dot(self.M,
                                                    Xddhp)  # f is positive in textbook but should most likely be negative
        return - np.linalg.solve(self.A(t), b)

    def Xddhp(self, X, Xd, Xdd):
        return -self.T1 * X - self.T3 * Xd - self.T4 * Xdd

    def Xdhp(self, X, Xd, Xdd):
        return -self.T2 * X + self.T5 * Xd + self.T6 * Xdd  # dt allready acounted for in constants function

    def Xddp(self, Xddhp, Xp):
        return Xddhp + self.T1 * Xp

    def Xdp(self, Xdhp, Xp):
        return Xdhp + self.T2 * Xp

    def A(self, t):
        return self.T1 * self.M + self.T2 * self.C + self.K(t)

    def newmark_solve(self):
        self.constants()

        sol_len = len(self.time_range)
        sol = np.zeros((self.dimensionality * 3, sol_len))

        Xdd = self.get_Xdd0()
        Xd = self.Xd0
        X = self.X0

        sol[:, 0] = np.vstack((X, Xd, Xdd))[:, 0]

        for time_step, time_index in zip(self.time_range[1:], range(1, sol_len)):
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


class LMM_sys(RungeKutta, Newmark_Beta, Stiff_DE):
    """
    Lumped mass model object for second order equation of Newtons second law
    Time variable stiffness, All other parameters are constant
    """

    def __init__(self, M, C, K, f, X0, Xd0, time_range):
        self.M = M
        self.M_inv = np.linalg.inv(M)
        self.C = C
        self.K = K
        self.X0 = X0
        self.Xd0 = Xd0
        self.f = f

        self.time_range = time_range

        self.dt = np.average(np.diff(self.time_range))
        self.dimensionality = np.shape(M)[0]

        self.dim = np.shape(self.M)[0]  # Matrix dimensionality

        return

    def solve_de(self, solver):
        if solver == "RK":
            return self.rk_solve()

        if solver == "Newmark":
            # {"beta_1": 0.5, "beta_2": 0.25}
            return self.newmark_solve()

        if solver == "Radau" or "BDF":
            return self.stiff_solve(solver)

        else:
            print("not implemented")

    def plot_solution(self, solution, state_time_der):

        nstate = int(np.shape(solution)[0] / 3)
        if state_time_der == "Displacement":
            start = 0

        if state_time_der == "Velocity":
            start = nstate * 1

        if state_time_der == "Acceleration":
            start = nstate * 2

        end = start + nstate

        lables = ("x_c",
                  "y_c",
                  "u_c",
                  "x_r",
                  "y_r",
                  "u_r",
                  "x_s",
                  "y_s",
                  "u_s",
                  "zeta_1",
                  "nu_1",
                  "u_1")

        # plt.figure(state_time_der)
        # plt.ylabel("Displacement [m]")
        # plt.xlabel("Time [s]")
        # p = plt.plot(self.time_range[1:], solution[1:, start:end])
        # plt.legend(iter(p), lables)

        plt.figure("Rotational DOF, carrier, sun, planet" + state_time_der)
        plt.plot(self.time_range, solution[:, start + 0], label="u_c")
        plt.plot(self.time_range, solution[:, start + 2], label="u_r")
        plt.plot(self.time_range, solution[:, start + 8], label="u_s")
        plt.plot(self.time_range, solution[:, start + 11], label="u_1")
        plt.legend()

        plt.figure("x-translation, carrier, sun, planet" + state_time_der)
        plt.plot(self.time_range, solution[:, start + 0], label="x_c")
        plt.plot(self.time_range, solution[:, start + 3], label="x_r")
        plt.plot(self.time_range, solution[:, start + 6], label="x_s")
        plt.plot(self.time_range, solution[:, start + 9], label="x_p1")
        plt.legend()

        plt.figure("Planet displacement")
        plt.plot(self.time_range, solution[:, start + 9], label="zeta_1")
        plt.plot(self.time_range, solution[:, start + 10], label="nu_1")
        plt.legend()

        # plt.plot(timerange, sol_nm[:, -3], label="Newmark")
        # plt.ylim(np.min(sol_rk[:, -3]),np.max(sol_rk[:, -3]))
