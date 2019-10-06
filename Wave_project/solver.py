from wave_project import first_time_step, fill_points_vectorized, fill_points_scalar
import numpy as np
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


def exact(x, y, t):
    return 1


def solver(b, T, Lx, Ly, dt, I, V, q, Nx, Ny):
    f = ode_source_term(exact, b, q)
    u, u_1, u_2, x, y, t = first_time_step(b, T, Lx, Ly, dt, f, I, V, q, Nx, Ny)
    u, u_1, u_2 = fill_points_vectorized(u, u_1, u_2, x, y, t)
