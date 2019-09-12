import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
A, B, V, t, I, w, dt = sym.symbols('A B V t I w dt')  # global symbols
f = None  # global variable for the source term in the ODE


def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)


def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = u(t + dt) + w**2*u(t)*dt**2 - f*dt**2 - 2*u(t) + u(t - dt)
    return sym.simplify(R)


def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = u(dt) - 0.5*f.subs(t, 0)*dt**2 + 0.5*w**2*I*dt**2 - V*dt - I
    return sym.simplify(R)


def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    num = (u(t+dt) - 2*u(t) + u(t-dt))/dt**2
    return sym.simplify(num)


def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    #N = 1000
    #dt = 1/(N+1)
    print('=== Testing exact solution: %s ===' % u)
    print("Initial conditions u(0)=%s, u'(0)=%s:" %
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0)))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print('residual step1:', residual_discrete_eq_step1(u))
    print('residual:', residual_discrete_eq(u))


def linear():
    return lambda t: V*t + I


def quadratic():
    return lambda t: B*t**2 + V*t + I


def cubic():
    return lambda t: A*t**3 + B*t**2 + V*t + I


def solver(I, w, dt, T, f, V):
    Nt = int(round(T/dt))
    u = np.zeros(Nt+1)
    t1 = np.linspace(0, Nt*dt, Nt+1)
    u[0] = I
    u[1] = f.subs(t, 0)*dt**2 - w**2*I*dt**2 + I + V*dt

    for i in tqdm(range(1, Nt)):
        u[i+1] = f.subs(t, dt*i)*dt**2 - w**2*u[i]*dt**2 + 2*u[i] - u[i-1]
    return u, t1


if __name__ == '__main__':
    # main(linear())
    main(quadratic())
    # main(cubic())
    #u_exact = quadratic()
    #f = ode_source_term(u_exact)
    # u, tt = solver(1, 1, 1/1000, 1,
    #              f.subs([(w, 1), (I, 1), (V, 1), (B, 1)]), 1)
    # print()
