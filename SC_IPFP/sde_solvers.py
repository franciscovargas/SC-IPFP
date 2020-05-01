import jax.numpy as np
import jax
import pylab as pl


key = jax.random.PRNGKey(0)
# key = None


def solve_sde_RK(alfa=None, beta=None, X0=None, dt=1.0, N=100, t0=0.0, DW=None,
                key = key):
    """
            Kloeden - Numerical Solution of stochastic differential
            equations (Springer 1992)  page XXX.
            Strong order 1.0 Runge Kutta scheme.
            http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_%28SDE%29
            dX = a(X,t)*dt + b(X, t)*dW
    Syntax:
    ----------
    solve_sde(alfa=None, beta=None, X0=None, dt=None, N=100, t0=0, DW=None)
    Parameters:
    ----------
        alfa  : a  function with two arguments, the X state and the time
                defines the differential equation.
        beta  : a  function with two arguments, the X state and the time
                defines the stochastic part of the SDE.
        X0    : Initial conditions of the SDE. Mandatory for SDEs
                with variables > 1 (default: gaussian np.random)
        dt    : The timestep of the solution
                (default: 1)
        N     : The number of timesteps (defines the length of the timeseries)
                (default: 100)
        t0    : The initial time of the solution
                (default: 0)
        DW    : The Wiener function in lambda notation
                (default: gaussian np.random number generator, \
                    [lambda Y, dt: randn(len(X0)) * np.sqrt(dt)] )
    Examples:
    ----------
    == Simple Wiener Process:
    dX = 0 + 1*dW
    alfa = lambda X,t: 0
    beta = lambda X,t: 1
    t, Y = solve_sde(alfa=alfa, beta=beta, dt=1, N=1000)
    == Stochastic Lorenz Equation:
    dX = s (Y - X) + Y * dW1
    dY = (r X - Y - X*Z) + dW2
    dZ = (X*Y - b Z)  + dW3
    xL = lambda X, t: 10.0 * (X[1] - X[0])  ;
    yL = lambda X, t: 28.0 * X[0] - X[1] - X[0] * X[2] ;
    zL = lambda X, t: X[0] * X[1] - 8.0/3.0 * X[2] ;
    alfa = lambda X, t: np.array( [xL(X,t), yL(X,t), zL(X,t)] );
    beta = lambda X, t: np.array( [     X[1],      1,      1] );
    X0 = [3.4, -1.3, 28.3];
    t, Y = solve_sde(alfa=alfa, beta=beta, X0=X0, dt=0.01, N=10000)
    
    
    """
    
    randn = lambda shape: jax.random.normal(key, shape=shape)
#     randn = np.random.randn
    
    if alfa is None or beta is None:
        raise ValueError("Error: SDE not defined.")
#     print(alfa(0, 0).shape)
#     import pdb; pdb.set_trace()
    X0 = randn(alfa(0, 0).shape) if X0 is None else np.array(X0)
    print(X0)
    DW = (lambda Y, dt: randn((len(X0),)) * np.sqrt(dt)) if DW is None else DW
    _, ti = np.zeros((N, len(X0))), np.arange(N)*dt + t0
    Y = X0.reshape(1,-1)
    _, Dn, Wn = X0, dt, 1

    for n in range(N-1):
        t = ti[n]
        a, b, DWn = alfa(Y[n, :], t), beta(Y[n, :], t), DW(Y[n, :], dt)
        # print Y[n,:]
        newY = Y[n, :] + a*Dn + b*DWn*Wn + \
                    0.5*(beta(Y[n, :] + b*np.sqrt(Dn), t) - b) * \
                    (DWn**2.0 - Dn)/np.sqrt(Dn)
#         print(Y.shape, newY.reshape(1,-1).shape)
        Y = np.concatenate((Y, newY.reshape(1,-1)), axis=0)
    return ti, Y
