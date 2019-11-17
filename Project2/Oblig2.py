from fenics import *
import numpy as np
import matplotlib.pyplot as plt
set_log_level(40)

%matplotlib inline

def error(u, u_e, V):
    
    v = interpolate(u_e, V)
    e = v.vector().get_local() - u.vector().get_local()
    err = np.sqrt(np.sum(e**2)/u.vector().get_local().size)

    return err

def boundary(x, on_boundary):
    return on_boundary

def solver(dim, N, f, alpha, I, rho, dt, T, u_e, u_D = None):
    
    num_steps = int(T/dt)
    
    if dim == 'interval':
        mesh = UnitIntervalMesh(N)
    elif dim == 'square':
        mesh = UnitSquareMesh(N, N)
    elif dim == 'box':
        mesh = UnitCubeMesh(N, N, N)
        
    V = FunctionSpace(mesh, 'P', 1)
    
    if u_D != None:
        bc = DirichletBC(V, u_D, boundary)
    
    u_n = interpolate(I, V)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = u*v*dx - dt/rho*div(alpha(u_n)*grad(u))*v*dx
    #a = u*v*dx + (dt/rho)*alpha(u_n)*dot(grad(u),grad(v))*dx
    L = u_n*v*dx + dt/rho*f*v*dx
    
    u = Function(V)
    t = 0
    
    err = np.zeros(num_steps)
    
    for i in range(num_steps):
        t += dt
        
        if u_D != None:
            solve(a==L, u, bc)
        else: 
            solve(a==L, u)
        
        u_e.t = t
        err[i] = error(u, u_e, V)
        
        u_n.assign(u)
    
    return u, err
    
def d():
    alpha = lambda u: 1.0
    I = Constant(1.0)
    f = Constant(0.0)
    rho = Constant(1.0)
    N = 10
    T = 1.0
    dt = 0.01
    dim = ['interval', 'square', 'box']
    
    for i in range(3):
        u, E = solver(dim[i], N, f, alpha, I, rho, dt, T, I)
        print('For the %s mesh with N = %i, the error:' % (dim[i], N), E[-1])

    
def e():
    alpha = lambda u: 1.0
    I = Expression('cos(pi*x[0])', degree=1)
    f = Constant(1.0)
    f = Expression('0', degree=1)
    rho = Constant(1.0)
    T = 1
    dt = 0.05
    u_D = Expression('0.0', degree = 2)
    u_e = Expression('exp(-pow(pi, 2)*t)*cos(pi*x[0])', t = 0, degree = 1)
    
    for i in range(5):
        N = int(1/dt)
        u, E = solver('square', N, f, alpha, I, rho, dt, T, u_e, u_D = u_D)
        
        E_ = E[-1]/dt
        print('h = %.5f, E/h = %.6f, mesh = [%i, %i]' % (dt,E_,N,N))
        
        dt = dt/2

def f():
    alpha = lambda u: 1 + u*u
    rho = 1.0
    I = Expression('0.0', degree = 1)
    f = Expression('-rho*pow(x[0], 3)/3 + rho*pow(x[0], 2)/2 + 8*pow(t, 3)*pow(x[0],7)/9 - 28*pow(t, 3)*pow(x[0],6)/9 +7*pow(t, 3)*pow(x[0], 5)/2 - 5*pow(t,3)*pow(x[0], 4)/4 + 2*t*x[0] - t', rho = rho, t = 0, degree = 1)
    u_e = Expression('t*pow(x[0], 2)*(0.5 - x[0]/3)', t = 0, degree = 1)
    dt = 0.01
    #times = np.arange(1,10,2)
    N = 10
    #times = np.arange(0.5, 4, 0.5)
    #times = np.arange(0.1, 1.1, 0.1)
    times = [0.1, 0.5, 1, 2, 5]
    for time in times:
        
        u_d = Expression('0', degree = 1)
    
        u, E = solver('interval', N, f, alpha, I, rho, dt, time, u_e, u_D = u_d)
        
        x = np.linspace(0, 1, 1001)
        x_ = np.linspace(0, 1, u.vector().get_local().size)
        t = time
        u_ex = t*x**2*(0.5 - x/3)
        plt.plot(x, u_ex, label = 'Exact solution')
        plt.plot(x_, u.vector().get_local()[::-1],'o', label = 'Numerical solution')
        plt.legend()
        plt.title('Results for T = %.1f'% time)
        plt.show()
    
    
if __name__ == '__main__':
    #d()
    #e()
    f()