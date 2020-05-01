import jax.numpy 
import autograd.numpy
from C_IPFP.sde_solvers import solve_sde_RK


# np = jax.numpy
np = autograd.numpy



class cIPFP(object):
    
    def __init__(self, X_0, X_1, batch_size=None,
                 number_time_steps=16, sde_solver=solve_sde_RK):
        pass