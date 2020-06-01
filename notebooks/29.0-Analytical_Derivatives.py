import sympy as sym

def one_dof_oscillator():
    sigma,t,a,omega,b = sym.symbols("sigma t a omega b",real=True)

    x = sym.exp(sigma*t)*(a*sym.cos(omega*t) + b*sym.sin(omega*t))

    xdot = sym.diff(x,t)
    print("xdot:", xdot)

    xdotdot = sym.diff(xdot,t)

    print("xdotdot",sym.simplify(xdotdot))
    return xdot, xdotdot


# For underdamped

sym.init_printing()

zeta,omega_n,t,d_0,v_0 = sym.symbols("zeta,omega_n,t,d_0,v_0")


omega_d = omega_n*sym.sqrt(abs(zeta**2-1))

a = sym.exp(-zeta*omega_n*t)
b = d_0*sym.cos(omega_d*t)
c = ((v_0+zeta*omega_n*d_0)/omega_d)*sym.sin(omega_d*t)

x = a*(b+c)   # For 0<= zeta <1
print("x")
sym.pretty_print(x)
print("")
print("")
print("")


xdot = sym.diff(x,t)
xdotdot = sym.diff(x,t)
print("xdotdot")
sym.pretty_print(xdotdot)

x = sym.Matrix([xdotdot])
y = sym.Matrix([zeta,omega_n,d_0,v_0])
jac = x.jacobian(y)

l = sym.latex(sym.trigsimp(xdotdot))

print(l)




