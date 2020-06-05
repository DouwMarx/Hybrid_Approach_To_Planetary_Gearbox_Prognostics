import numpy as np
import scipy.integrate as inter
import matplotlib.pyplot as plt


class SimpleHarmonicOscillator(object):
    def __init__(self):
        self.m = 0.1  # Frequency
        self.c = 0.6  # Damping Ratio
        self.F = 10  # Peak height, Vibration magnitude

        self.delta_k = 10
        self.k_mean = 20

        self.X0 = np.array([self.F / self.k_mean, 0])
        # self.X0 = np.array([self.F/(self.k_mean+self.delta_k),0])

        self.t_range = np.linspace(0, 5, 1000)

        self.t_step_start = 2
        self.t_step_duration = 0.1

    def Xdot(self, t, X):
        E = np.array([[0, 1], [-self.k(t) / self.m, -self.c / self.m]])
        Q = np.array([[0], [self.F / self.m]])

        return np.dot(E, X) + Q

    def Xdotdot(self, y):
        """

        Parameters
        ----------
        y: Solution to the integrated differential equation

        Returns
        -------

        """
        Xdotdot = np.zeros((2, len(self.t_range)))
        for t, i in zip(self.t_range, range(len(self.t_range))):
            E = np.array([[0, 1], [-self.k(t) / self.m, -self.c / self.m]])
            Q = np.array([[0], [self.F / self.m]])
            Xdd = np.dot(E, np.array([y[:, i]]).T) + Q
            Xdotdot[:, i] = Xdd[:, 0]

        return Xdotdot.T[:, 1]

    def k(self, t):
        # variant = "square_mean_delta"
        variant = "sine_mean_delta_drop"
        # variant = "sine_mean_delta_step"
        #variant = "square_mean_delta"
        if variant == "square_delta":
            if t <= self.t_step_start:
                return 0

            elif t <= self.t_step_start + self.t_step_duration:
                return self.delta_k

            else:
                return 0

        if variant == "square_mean_delta":
            if t <= self.t_step_start:
                return self.k_mean

            elif t <= self.t_step_start + self.t_step_duration:
                return self.k_mean + self.delta_k

            else:
                return self.k_mean

        if variant == "sine_delta":
            if t <= self.t_step_start:
                return 0  # self.k_mean

            elif t <= self.t_step_start + self.t_step_duration:
                return (np.cos(0.5 * 2 * np.pi * t) + 1) * self.delta_k  # + self.k_mean

            else:
                return 0  # self.k_mean

        if variant == "sine_mean_delta":
            if t <= self.t_step_start:
                return 0 + self.k_mean

            elif t <= self.t_step_start + self.t_step_duration:
                return (-np.cos(
                    2 * np.pi * (t - self.t_step_start) / self.t_step_duration) + 1) * self.delta_k + self.k_mean

            else:
                return 0 + self.k_mean

        if variant == "sine_mean_delta_drop":
            if t <= self.t_step_start:
                return self.delta_k + self.k_mean

            elif t <= self.t_step_start + self.t_step_duration:
                return 0.5 * (np.cos(
                    np.pi * (t - self.t_step_start) / self.t_step_duration) + 1) * self.delta_k + self.k_mean

            else:
                return 0 + self.k_mean

        if variant == "sine_mean_delta_step":
            if t <= self.t_step_start:
                return self.k_mean

            elif t <= self.t_step_start + self.t_step_duration:
                return 0.5 * (-np.cos(
                    np.pi * (t - self.t_step_start) / self.t_step_duration) + 1) * self.delta_k + self.k_mean

            else:
                return self.delta_k + self.k_mean

    def integrate_ode(self):
        sol = inter.solve_ivp(self.Xdot, [self.t_range[0], self.t_range[-1]],
                              self.X0,
                              vectorized=True,
                              t_eval=self.t_range)
        return sol


sho1 = SimpleHarmonicOscillator()

sol = sho1.integrate_ode()

plt.figure("Displacement")
plt.plot(sol["t"], sol["y"][0, :])

plt.figure("Velocity")
plt.plot(sol["t"], sol["y"][1, :])

plt.figure("Acceleration")
plt.plot(sol["t"], sho1.Xdotdot(sol["y"]))

klist = []
for t in sho1.t_range:
    klist.append(sho1.k(t))

plt.figure()
plt.plot(klist)
