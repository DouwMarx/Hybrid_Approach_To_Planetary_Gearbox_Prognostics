import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import dill
import definitions
import scipy.signal as scisig
import scipy.optimize as opt
import scipy.integrate as inter

def make_1dof_spring_mass_system(plot=False):
    zeta,omega_n,t,d_0,v_0 = sym.symbols("zeta,omega_n,t,d_0,v_0",real=True)
    omega_d = omega_n*sym.sqrt(abs(zeta**2-1)) # Damped natural frequency

    a = sym.exp(-zeta*omega_n*t)
    b = d_0*sym.cos(omega_d*t)
    c = ((v_0+zeta*omega_n*d_0)/omega_d)*sym.sin(omega_d*t)

    x = a*(b+c)   # For 0<= zeta <1
    xdot = sym.diff(x,t)
    xdotdot = sym.diff(xdot,t)
    #sym.pretty_print(xdotdot)

    #function_to_compute_jacobian_for = sym.Matrix([xdotdot])
    #variables_to_derive_for = sym.Matrix([zeta,omega_n,d_0,v_0])
    #jac = sym.trigsimp(function_to_compute_jacobian_for.jacobian(variables_to_derive_for))
    dxdd_dzeta = sym.diff(xdotdot,zeta)
    dxdd_domega_n = sym.diff(xdotdot,omega_n)
    dxdd_dd_0 = sym.diff(xdotdot,d_0)
    dxdd_dv_0 = sym.diff(xdotdot,zeta)
    #
    jac = np.array([dxdd_dzeta,dxdd_domega_n,dxdd_dd_0,dxdd_dv_0])


    xdotdot_func = sym.lambdify([zeta,omega_n,t,d_0,v_0],xdotdot,"numpy")
    jac_func = sym.lambdify([zeta,omega_n,t,d_0,v_0],jac,"numpy")

    if plot:
        t_range = np.linspace(0,1,1000)
        testsol = xdotdot_func(0.01,10*2*np.pi,t_range,1,1)
        plt.figure()
        plt.plot(t_range, testsol)

    # Get the latex expression for reporting
    xdotdot_latex = sym.latex(xdotdot)
    jac_latex = sym.latex(jac)

    return xdotdot_func, jac_func, xdotdot_latex,jac_latex

def save_sympy_lambda_funcs():
    xddf,jf,xddlx,jlx = make_1dof_spring_mass_system(plot=False)

    # Save lambdified function
    dill.settings["recurse"] = True
    fname = "xdd"
    with open(definitions.root + "\\models\\sympy_lambda_funcs" + "\\" + fname + ".sym_func", 'wb') as config:
        dill.dump(xddf, config)

    fname = "jac"
    with open(definitions.root + "\\models\\sympy_lambda_funcs" + "\\" + fname + ".sym_func", 'wb') as config:
        dill.dump(jf, config)

    return

def ratio_of_d0():
    xdotdot_2 = xdotdot_1.subs(d_0,n*d_0)
    # subbed_t = xdotdot.subs(t,0)
    # print(subbed_t)
    # v_0_sol = sym.solve(subbed_t, v_0)
    # print(v_0_sol[0].simplify())
    # #print(sym.latex(subbed_t.simplify()))
    print("Ratio between responses with ratio n between d_0 and d_1")
    ratio = xdotdot_2/xdotdot_1
    print(ratio.simplify())

    print("")
    print("at t=0 and v=0")
    ratio = ratio.subs(t,5)
    ratio = ratio.subs(v_0,0)
    print(ratio.simplify())
    #sol = sym.solve(ratio,n)
    return

class one_dof_sys(object):
    """ Used to fit 1DOF spring mass damper system to Acceleration signal"""

    def __init__(self, acc_sig, fs):
        self.fs = fs
        self.acc = acc_sig
        self.trans_len = len(acc_sig)

        strategy = "fit_sdof_sol"
        if strategy == "fit_sdof_sol":
            #self.xdd_func = an_sdof.xddf
            self.trange = np.linspace(0,self.trans_len/self.fs,self.trans_len)
            self.xdd_detrend = self.detrend_sig(self.acc)

            fname = "xdd"
            with open(definitions.root + "\\models\\sympy_lambda_funcs" + "\\" + fname + ".sym_func", 'rb') as config:
                self.xdd_func = dill.load(config)

            fname = "jac"
            with open(definitions.root + "\\models\\sympy_lambda_funcs" + "\\" + fname + ".sym_func", 'rb') as config:
                self.jac_func = dill.load(config)

        if strategy == "integration":
            self.vel, self.disp = self.integrate_signal(self.acc, self.fs, int_type="frequency")
            self.delta_k = 1e6
            self.acc = self.acc - np.mean(self.acc)
            self.vel = self.vel - np.mean(self.vel)
            self.disp = self.disp - np.mean(self.disp)

    def detrend_sig(self,acc_sig,plot=False):


        detrended = scisig.detrend(acc_sig)
        if plot:
            plt.figure()
            plt.plot(acc_sig,label="Original")
            plt.plot(detrended,label="Detrended")
            plt.legend()

        return detrended

    def freq_int(self, x, fs, ints):
        "Performs frequency intergration"
        h = np.fft.fft(x)
        n = len(h)

        w = np.fft.fftfreq(n, d=1 / fs) * 2 * np.pi
        w[0] = 1

        w = w * 1j

        g = np.divide(h, w ** ints)
        y = np.fft.ifft(g)
        y = np.real(y)
        return (y)

    def integrate_signal(self, acc_sig, fs, int_type="time"):
        """
        Performs either time of frequency domain integration
        Parameters
        ----------
        acc_sig
        fs
        int_type

        Returns
        -------

        """
        if int_type == "time":
            vel = inter.cumtrapz(acc_sig, None, dx=1 / fs)
            disp = inter.cumtrapz(vel, None, dx=1 / fs)

            return vel, disp

        if int_type == "frequency":
            vel = self.freq_int(acc_sig, fs, 1)
            disp = self.freq_int(acc_sig, fs, 2)
            return vel, disp

    def plot_integration_test(self):
        fig, axs = plt.subplots(2, 3)
        i = np.random.randint(np.shape(trans)[0])
        acc = trans[i, :]
        for int_type, i in zip(["time", "frequency"], [0, 1]):
            axs[i, 0].plot(acc, label="acc")
            vel, disp = self.integrate_signal(acc, 1, int_type=int_type)

            axs[i, 1].plot(vel, label="vel")

            axs[i, 2].plot(disp, label="disp")
        return

    def cost(self, theta):
        """
        Cost funcion for determining model parameters
        Parameters
        ----------
        #theta = [m,c,k,F]
        theta = [m,c,F]

        Returns
        -------

        """
        # return np.linalg.norm(theta[0]*self.acc + theta[1]*self.disp + theta[2]*self.vel - theta[3])

        return np.linalg.norm(theta[0] * self.acc + self.delta_k * self.disp + theta[1] * self.vel - theta[2])

    def cost_ana_sol(self, theta):
        """
        Cost funcion for determining model parameters
        Parameters
        ----------
        theta = [zeta,omega_n,d_0,v_0]]

        Returns
        -------

        """

        candidate = self.xdd_func(theta[0],theta[1],self.trange,theta[2],theta[3])

        return np.linalg.norm(self.xdd_detrend - candidate)


    def run_optimisation(self):
        # bnds = ((0, None), (0, None),(0,None),(0,None))
        # s = opt.minimize(self.cost,np.ones(4),bounds=bnds)

        bnds = np.array([[1, 100],  # m
                         [1, 100],  # c
                         # [1,100],  #k
                         [1, 100]])  # F

        s = opt.differential_evolution(self.cost, bounds=bnds)
        return s

    def fun(self,theta,t,y):
        return self.xdd_func(theta[0],theta[1],t,theta[2],theta[3]) - y

    def jac(self,theta,t,y):
        J = np.empty((len(t),len(theta)))
        jac_eval = self.jac_func(theta[0],theta[1],t,theta[2],theta[3])
        J[:,0] = jac_eval[0]
        J[:,1] = jac_eval[1]
        J[:,2] = jac_eval[2]
        J[:,3] = jac_eval[3]
        return J

    def do_least_squares(self,plot=False):

        theta0 = np.array([0.1,5000*np.pi*2,0,0])
        bnds = ([0,1000,0,0],[1,np.inf,1,1])  # [zeta,omega,d0,v0]
        sol = opt.least_squares(self.fun,
                                theta0,
                                jac=self.jac,
                                bounds= bnds,#(0,np.inf),
                                args=(self.trange,self.xdd_detrend),
                                xtol = 1e-12, # Tollerance on independent variables exit status 3, None disables it
                                ftol = 1e-12, # exit condition 2
                                #x_scale="jac", #Updates scale according to jacobian
                                x_scale=np.array([1e-1,1e3,1e-6,1e-8]),
                                loss="linear", # Loss function to use (default = "linear" for regular leastsq)
                                verbose=0)
        if plot:
            plt.figure()
            plt.scatter(self.trange,self.xdd_detrend)
            theta = sol["x"]
            model = self.xdd_func(theta[0],theta[1],self.trange,theta[2],theta[3])
            plt.plot(self.trange,model)
            plt.xlim(0,self.trange[-1])
        return sol


    def run_optimisation_ana_sol(self, plot=False):
        bnds = ((0, None), (0, None),(0,None),(0,None))
        startpoint = np.array([0.01,3000*np.pi*2,0,0])
        s = opt.minimize(self.cost_ana_sol,
                         startpoint,
                         bounds=bnds,
                         tol=1)

        #bnds = np.array([[1, 100],  # m
        # [1, 100],  # c
        # # [1,100],  #k
        # [1, 100]])  # F
        #
        #s = opt.differential_evolution(self.cost, bounds=bnds)

        if plot:
            plt.figure()
            plt.scatter(self.trange,self.xdd_detrend)
            theta = s["x"]
            model = self.xdd_func(theta[0],theta[1],self.trange,theta[2],theta[3])
            plt.plot(self.trange,model)
            plt.xlim(0,self.trange[-1])
        return s

#dum,dum,latex,dum = make_1dof_spring_mass_system()
#print(latex)
#
# zeta,omega_n,t,d_0,n,v_0 = sym.symbols("zeta,omega_n,t,d_0,n,v_0",real=True)
# omega_d = omega_n*sym.sqrt(-(zeta**2-1)) # Damped natural frequency, - because 0<- zeta <1
#
# a = sym.exp(-zeta*omega_n*t)
# b = d_0*sym.cos(omega_d*t)
# c = ((v_0+zeta*omega_n*d_0)/omega_d)*sym.sin(omega_d*t)
#
# x = a*(b+c)   # For 0<= zeta <1
# xdot = sym.diff(x,t)
# xdotdot_1 = sym.diff(xdot,t)
#
#
#
# xdotdot_2 = xdotdot_1.subs(v_0,n*v_0)
# # subbed_t = xdotdot.subs(t,0)
# # print(subbed_t)
# # v_0_sol = sym.solve(subbed_t, v_0)
# # print(v_0_sol[0].simplify())
# # #print(sym.latex(subbed_t.simplify()))
# print("Ratio between responses with ratio n between d_0 and d_1")
# ratio = xdotdot_2/xdotdot_1
# print(ratio.simplify())
#
# print("")
# print("at t=0 and d=0")
# ratio = ratio.subs(t,0)
# ratio = ratio.subs(d_0,0)
# print(ratio.simplify())
# #sol = sym.solve(ratio,n)
