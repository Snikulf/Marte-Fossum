import numpy as np
from wave_project import first_time_step, fill_points_vectorized, fill_points_scalar
import sympy as sym


def ode_source_term(u, b, q):

    x, y, t = sym.symbols("x y t")

    utt = sym.diff(u(x, t, y), t, t)
    ut = sym.diff(u(x, y, t), t)
    ux = sym.diff(u(x, y, t), x)
    dxqux = sym.diff(q(x, y) * ux, x)
    uy = sym.diff(u(x, y, t), y)
    dyquy = sym.diff(q(x, y) * uy, y)

    f = utt + b * ut - dxqux - dyquy

    return f


def exact(x, y, t, Lx=1, Ly=1, mx=1, my=1):
    A = 1
    kx = mx * np.pi / Lx
    ky = my * np.pi / Ly
    omega = 1

    return A * sym.cos(kx * x) * sym.cos(ky * y) * sym.cos(omega * t)


def solver(b, T, Lx, Ly, dt, q, Nx, Ny):
    x = sym.symbols("x")
    y = sym.symbols("y")
    t = sym.symbols("t")
    f = sym.lambdify((x, y, t), ode_source_term(exact, b, q))
    I = sym.lambdify((x, y), exact(x, y, t).subs(t, 0))
    V = sym.lambdify((x, y), sym.diff(exact(x, y, t), t).subs(t, 0))
    u, u_1, u_2, x_tmp, y_tmp, t_tmp = first_time_step(
        b, T, Lx, Ly, dt, f, I, V, q, Nx, Ny
    )
    u, u_1, u_2 = fill_points_vectorized(
        u, u_1, u_2, x_tmp, y_tmp, t_tmp, b, I, V, f, q, Nx, Ny
    )


"""
    u_exact1 = sym.lambdify((x, y), exact(x, y, t).subs(t, T))
    u_ex1 = u_exact1(x_tmp, y_tmp)
    dx = x_tmp[1] - x_tmp[0]
    dy = y_tmp[1] - y_tmp[0]
    e1 = u_ex1 - u
    L1 = np.sqrt(dx * dy * np.sum(e1 ** 2))

    u_exact2 = sym.lambdify((x, y), exact(x, y, t).subs(t, T - dt))
    u_ex2 = u_exact2(x_tmp, y_tmp)
    e2 = u_ex2 - u_1
    L2 = np.sqrt(dx * dy * np.sum(e2 ** 2))

    u_exact3 = sym.lambdify((x, y), exact(x, y, t).subs(t, T - 2 * dt))
    u_ex3 = u_exact2(x_tmp, y_tmp)
    e3 = u_ex3 - u_1
    L3 = np.sqrt(dx * dy * np.sum(e3 ** 2))

    r1 = np.log(L2 / L1) / np.log(2)
    r2 = np.log(L3 / L2) / np.log(2)
 """


Nx = 20
Ny = 20
q = lambda x, y: 0  # np.zeros((Nx + 1, Ny + 1))
solver(0, 1, 1, 1, 0.01, q, Nx, Ny)
