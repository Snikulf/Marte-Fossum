import sympy as sym
V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
f = None  # global variable for the source term in the ODE
B = 1


def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)


def residual_discrete_eq(u, dt):
    """Return the residual of the discrete eq. with u inserted."""
    #f = ode_source_term(u)
    R = u(t + dt) - w**2*u(t) + f + 2*u(t) - u(t - dt)
    return sym.simplify(R)


def residual_discrete_eq_step1(u, dt):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    #f = ode_source_term(u)
    R = u(dt) - (f.subs(t, 0)*dt**2 - w*I*dt**2 + V*dt + I)
    return sym.simplify(R)


def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    num = (u(t+dt) - 2*u(t) + u(t-dt))/dt**2
    return num


def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    N = 1000
    dt = 1/(N+1)
    print('=== Testing exact solution: %s ===' % u)
    print("Initial conditions u(0)=%s, u'(0)=%s:" %
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0)))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print('residual step1:', residual_discrete_eq_step1(u, dt))
    print('residual:', residual_discrete_eq(u, dt))


def linear():
    main(lambda t: V*t + I)


def quadratic():
    main(lambda t: B*t**2 + V*t + I)


if __name__ == '__main__':
    linear()
    # quadratic()
