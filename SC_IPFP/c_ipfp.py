import jax.numpy  as np
# import autograd.numpy
import numpy as np
from C_IPFP.sde_solvers import solve_sde_RK


# np = jax.numpy
# np = autograd.numpy



class cIPFP(object):
    
    def __init__(self, X_0, X_1, batch_size=None, b_forward, b_backward
                 number_time_steps=16, sde_solver=solve_sde_RK, sigma_sq=1):
        
        self.solver = sde_solver
        
        self.number_time_steps = number_time_steps
        self.dt = 1.0 / number_time_steps
        
        self.batch_size = batch_size
        
        self.X_0 = X_0
        self.X_1 = X_1
        
        self.b_forward = b_forward
        self.b_backward = b_backward
        
        
        self.sigma = lambda x,t: sigma_sq
    
    @staticmethod
    def loss_for_trajectory(Xt, b_f, b_b, dt, forwards = True):
        b_minus  = b_b(Xt)
        b_plus = b_f(Xt)
        
        delta_Xt = Xt[:-1, :]  - Xt[1:, :]
        
        sign = 1.0 if forwards else -1.0
        
        ito_integrals = sign *  (b_plus[1:,:] - b_minus[:-1,:])  * delta_Xt
        
        time_integral = sign *  (b_plus**2 - b_minus**2) * dt
        
        return ito_integrals.sum() - 0.5 * time_integral.sum()
        
        
    def sample_trajectory(self, X, forwards=True):
        
        # backwards discretisation has a sign flip         
        b = self.b_forward if forward else (lambda x,t, theta: -self.b_backward(x,t,theta))
        
        return self.sde_solver(alfa=b, beta=self.sigma,
                               dt=self.dt, X0=X,
                               N=self.number_time_steps)
    
    def inner_loss(self, theta, forwards=True):
        
        X = self.X_0 if forward else self.X_1
               
        J = 0
        
        for x in X:
            t, Xt = self.sample_trajectory(x, forwards=forwards)
            
            J += loss_for_trajectory(Xt, self.b_forward, self.b_backward, dt, forwards=forwards)
        
        J /= len(X)
        
        return J
            
        