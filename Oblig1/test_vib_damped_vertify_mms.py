from vib_undamped_verify_mms import *


def test_solver_1():
    def u_exact(t): return V*t + I
    f = ode_source_term(u_exact)
    epsilon = 1e-5

    u, tt = solver(1, 1, 1/1000, 1, f.subs([(I, 1), (w, 1), (V, 1)]), 1)

    exact_values = np.array(
        list(map(lambda x: x.subs([(I, 1), (V, 1)]), u_exact(tt))))

    assert all(abs(exact_values-u) < epsilon)


def test_solver_2():
    def u_exact(t): return B*t**2 + V*t + I
    f = ode_source_term(u_exact)
    epsilon = 1e-2

    u, tt = solver(1, 1, 1/1000, 1,
                   f.subs([(I, 1), (w, 1), (V, 1), (B, 1)]), 1)

    exact_values = np.array(
        list(map(lambda x: x.subs([(I, 1), (V, 1), (B, 1)]), u_exact(tt))))
    assert all(abs(exact_values-u) < epsilon)
