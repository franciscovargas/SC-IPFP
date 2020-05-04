import jax.numpy as np
import numpy as onp
import jax
import pylab as pl
from jax import jit
from functools import partial

key = jax.random.PRNGKey(0)
# key = None


@partial(jit, static_argnums=(0,1, 4))
def inner_jit(alfa, beta, Y, ti, N, Dn, DWs, Wn, dt, theta):
    alfa_ = alfa
    if theta is not None:
        alfa_ = lambda X,t: alfa(theta, X)
    for n in range(N-1):
        t = ti[n]
        a, b, DWn = alfa_(Y[n, :], t), beta(Y[n, :], t), DWs[n,:]
        # print Y[n,:]
        newY = (  
            Y[n, :] + a * Dn + b * DWn * Wn + 
            0.5 * ( beta(Y[n, :] + b * np.sqrt(Dn), t) - b ) * 
            (DWn**2.0 - Dn) / np.sqrt(Dn)
        )

        Y = jax.ops.index_update(Y, jax.ops.index[n+1,:],  newY)
    return ti, Y


# @jit
def solve_sde_RK(alfa=None, beta=None, X0=None, dt=1.0, N=100, t0=0.0,
                key = key, theta=None):
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
    
    """
    
    randn = lambda shape: jax.random.normal(key, shape=shape)
    randn = onp.random.randn

       
    if alfa is None or beta is None:
        raise ValueError("Error: SDE not defined.")

        
    X0 = randn(*alfa(0, 0).shape) if X0 is None else np.array(X0)
#     DW = (lambda Y, dt: randn((len(X0))) * np.sqrt(dt)) if DW is None else DW
    DWs  = randn(N-1, len(X0))  * np.sqrt(dt)
    
    
    
    Y, ti = np.zeros((N, len(X0))), np.arange(N)*dt + t0
    Y = jax.ops.index_update(Y, jax.ops.index[0,:],  X0)
    
    Dn, Wn = dt, 1
        
    return inner_jit(alfa, beta, Y, ti, N, Dn, DWs, Wn, dt, theta)
