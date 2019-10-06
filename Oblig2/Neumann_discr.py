from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sys

omega = 1
L = 1
"""
x, t = sym.symbols('x t')


def ode_cource_term(u):
    return sym.diff(u(x, t), t, t) - sym.diff(q(x)*sym.diff(u(x, t), x), x)


def q(x):
    return 1 + (x - L/2)**4


def exact(x, t):
    return sym.cos(sym.pi*x/L)*sym.cos(omega*t)


f = ode_cource_term(exact)
T = 1
Nt = 100
Nx = 50
dt = T/Nt
dx = L/Nx
tt = np.linspace(0, T, Nt+1)
xx = np.linspace(0, L, Nx+1)
u = np.zeros(len(xx), dtype=np.float64)
u_1 = np.zeros(len(u), dtype=np.float64)

# First time step

for i in range(1, Nx):
    u[i] = u_1[i] + 0.5*dt**2/dx**2 * (0.5*(q(xx[i]) + q(xx[i+1]))*(u_1[i+1] - u_1[i]) - 0.5*(
        q(xx[i]) + q(xx[i-1]))*(u_1[i] - u_1[i-1])) + dt**2*f.subs([(x, xx[i]), (t, tt[0])])

# Boundary values for x = 0, t = 0

u[0] = u_1[0] + 0.5*(dt/dx)**2*(0.5*(q(xx[0]) + q(xx[1]))*(u_1[1] - u_1[0]) -
                                0.5*(q(xx[0]) + q(xx[1]))*(u_1[0] - u_1[1])) + dt**2*f.subs([(x, xx[0]), (t, tt[0])])

# Boundary values x = L, t = 0

u[-1] = u_1[-1] + 0.5*(dt/dx)**2*(0.5*(q(xx[-1]) + q(xx[-2]))*(u_1[-2] - u_1[-1] - 0.5*(
    q(xx[-1]) + q(xx[-2]))*(u_1[-1] - u_1[-2])) + dt**2*f.subs([(x, xx[-1]), (t, tt[0])]))

u_2 = u_1
u_1 = u
error = []


for i in tqdm(range(1, Nt)):
    for j in range(1, Nx):
        u[j] = - u_2[j] + 2*u_1[j] + (dt/dx)**2*(0.5*(q(xx[j]) + q(xx[j+1])*(
            u_1[j+1] - u_1[j])) - 0.5*(q(xx[j]) + q(xx[j-1])*(u_1[j] - u_1[j-1]))) + dt**2*f.subs([(x, xx[j]), (t, tt[i])])

        # Boundary values x = 0

        u[0] = - u_2[0] + 2*u_1[0] + (dt/dx)**2*(0.5*(q(xx[0]) + q(xx[1]))*(
            u_1[1] - u_1[0]) - 0.5*(q(xx[0]) + q(xx[1]))*(u_1[0] - u_1[1])) + dt**2*f.subs([(x, xx[0]), (t, tt[i])])

        # Boundary values x = L

        u[-1] = - u_2[-1] + 2*u_1[-1] + (dt/dx)**2*(0.5*(q(xx[-1]) + q(xx[-2]))*(
            u_1[-2] - u_1[-1]) - 0.5*(q(xx[-1]) + q(xx[-2]))*(u_1[-1] - u_1[-2])) + dt**2*f.subs([(x, xx[-1]), (t, tt[i])])

        u_2 = u_1
        u_1 = u

    e = np.cos(np.pi*xx/L)*np.cos(omega*tt[i]) - u
    L = np.sqrt(dt*np.sum(e**2))
    error.append(L)

r = []

for i in range(10, len(error)):
    r.append(np.log(error[i-1]/error[i])/np.log(2))

print(r)


x1 = np.linspace(0, L, 1001)
eksakt = np.zeros(len(x1))

for i in range(len(x1)):
    eksakt[i] = float(exact(x, t).subs([(x, x1[i]), (t, T)]))

#plt.plot(xx, u)
#plt.plot(x1, eksakt)
#plt.legend(['numerisk', 'analytisk'])
# plt.show()

"""
omega = 1
L = 1


def q(x):
    return 1 + (x - L/2)**4


def exact(x, t):
    y = np.cos(np.pi*x/L)*np.cos(omega*t)
    print(y)
    return y


def f(x, t):
    del1 = - omega**2*np.cos(np.pi*x/L)*np.cos(omega*t)
    del2 = (1 + (x-L/2)**4)*np.cos(np.pi*x/L)*np.cos(omega*t)*(np.pi/L)**2
    del3 = np.pi/L*np.sin(np.pi*x/L)*np.cos(omega*t)*4*(x-L/2)**3
    return del1 + del2 + del3


T = 1
Nt = int(1e1)
Nx = int(1e2)
dt = T/Nt
dx = L/Nx
t = np.linspace(0, T, Nt+1)
x = np.linspace(0, L, Nx+1)
u = np.zeros(Nx+1, dtype=np.float64)
u_1 = exact(x, 0)

# First time step

for i in range(1, Nx):
    u[i] = 2*u_1[i] - u_1[i] + dt**2/dx**2 * (0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i]) - 0.5*(
        q(x[i]) + q(x[i-1]))*(u_1[i] - u_1[i-1])) + dt**2*f(x[i], t[1])

# Boundary values for x = 0

u[0] = u_1[0] + 0.5*(dt/dx)**2*(0.5*(q(x[0]) + q(x[1]))*(u_1[1] - u_1[0]) -
                                0.5*(q(x[0]) + q(x[1]))*(u_1[0] - u_1[1])) + dt**2*f(x[0], t[0])

# Boundary values x = L

u[-1] = u_1[-1] + 0.5*(dt/dx)**2*(0.5*(q(x[-1]) + q(x[-2]))*(u_1[-2] - u_1[-1]) -
                                  0.5*(q(x[-1]) + q(x[-2]))*(u_1[-1] - u_1[-2])) + dt**2*f(x[-1], t[0])

# Filling inner points

u_2 = u_1
u_1 = u
error = []

for i in range(1, Nt+1):

    u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + (dt/dx)**2*(0.5*(q(x[1:-1]) + q(x[2:])*(
        u_1[2:] - u_1[1:-1])) - 0.5*(q(x[1:-1]) + q(x[:-2])*(u_1[1:-1] - u_1[:-2]))) + dt**2*f(x[1:-1], t[i])

    # Boundary values x = 0

    u[0] = - u_2[0] + 2*u_1[0] + (dt/dx)**2*(0.5*(q(x[0]) + q(x[1]))*(
        u_1[1] - u_1[0]) - 0.5*(q(x[0]) + q(x[1]))*(u_1[0] - u_1[1])) + dt**2*f(x[0], t[i])

    # Boundary values x = L

    u[-1] = - u_2[-1] + 2*u_1[-1] + (dt/dx)**2*(0.5*(q(x[-1]) + q(x[-2]))*(
        u_1[-2] - u_1[-1]) - 0.5*(q(x[-1]) + q(x[-2]))*(u_1[-1] - u_1[-2])) + dt**2*f(x[-1], t[i])

    e = exact(x, t[i]) - u
    L = np.sqrt(dx*np.sum(e**2))
    error.append(L)

    u_2 = u_1
    u_1 = u


r = []

for i in range(len(error)):
    r.append(np.log(error[i-1]/error[i])/np.log(2))

print(r)


#plt.plot(x, u)
#plt.plot(x, exact(x, T))
#plt.legend(['numerisk', 'eksakt'])
# plt.axis([0, 1, -1, 1])
# plt.show()
