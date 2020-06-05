import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import time
import definitions
import pickle

def one_dof_oscillator():
    sigma,t,a,omega,b = sym.symbols("sigma t a omega b",real=True)

    x = sym.exp(sigma*t)*(a*sym.cos(omega*t) + b*sym.sin(omega*t))

    xdot = sym.diff(x,t)
    print("xdot:", xdot)

    xdotdot = sym.diff(xdot,t)

    print("xdotdot",sym.simplify(xdotdot))
    return xdot, xdotdot


# For underdamped spring mass system

def make_1dof_spring_mass_system(plot=False):
    zeta,omega_n,t,d_0,v_0 = sym.symbols("zeta,omega_n,t,d_0,v_0")
    omega_d = omega_n*sym.sqrt(abs(zeta**2-1)) # Damped natural frequency

    a = sym.exp(-zeta*omega_n*t)
    b = d_0*sym.cos(omega_d*t)
    c = ((v_0+zeta*omega_n*d_0)/omega_d)*sym.sin(omega_d*t)

    x = a*(b+c)   # For 0<= zeta <1
    xdot = sym.diff(x,t)
    xdotdot = sym.trigsimp(sym.diff(xdot,t))
    #sym.pretty_print(xdotdot)

    function_to_compute_jacobian_for = sym.Matrix([xdotdot])
    variables_to_derive_for = sym.Matrix([zeta,omega_n,d_0,v_0])
    jac = sym.trigsimp(function_to_compute_jacobian_for.jacobian(variables_to_derive_for))

    xdotdot_func = sym.lambdify([zeta,omega_n,t,d_0,v_0],xdotdot,"numpy")

    tstart = time.time()
    jac_func = sym.lambdify([zeta,omega_n,t,d_0,v_0],jac,"numpy")
    print("jac_eval",time.time()-tstart)

    tstart = time.time()
    if plot:
        t_range = np.linspace(0,1,1000)
        testsol = xdotdot_func(0.01,10*2*np.pi,t_range,1,1)
        plt.figure()
        plt.plot(t_range, testsol)

    print("eval_time",time.time()-tstart)

    # Get the latex expression for reporting
    xdotdot_latex = sym.latex(xdotdot)
    jac_latex = sym.latex(jac)

    return xdotdot_func, jac_func, xdotdot_latex,jac_latex

#xddf,jf,xddlx,jlx = make_1dof_spring_mass_system(plot=False)

def delta_k(m,x0,xd0,omega_n,zeta,xdd0):
    top = m*xdd0 -zeta*2*omega_n*m*xd0
    bot = x0
    right = omega_n**2 * m
    return top/bot - right

full = delta_k(0.153,2.7263e-7,2.92246e-4,1.8183e4,5.28378e-2,121/9.8)
half = delta_k(0.104,2.8722-7,2.832-4,1.8709e4,4.19341e-2,139/9.8)




print(full)
print(half)