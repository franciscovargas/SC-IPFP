import jax.numpy  as np
# import autograd.numpy
import numpy as np
from C_IPFP.sde_solvers import solve_sde_RK

from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
import numpy.random as npr

# np = jax.numpy
# np = autograd.numpy



class cIPFP(object):
    
    def __init__(self, X_0, X_1, weights=[100], batch_size=None, rng = npr.RandomState(0), 
                 number_time_steps=16, sde_solver=solve_sde_RK, sigma_sq=1, 
                step_size = 0.001, num_epochs = 10, momentum_mass = 0.9, create_network=False):
        
        self.solver = sde_solver
        
        self.number_time_steps = number_time_steps
        self.dt = 1.0 / number_time_steps
        
        
        self.batch_size_f = X_0.shape[0] if batch_size is None else batch_size
        self.batch_size_b = X_1.shape[0] if batch_size is None else batch_size
        
        
        self.X_0 = X_0
        self.X_1 = X_1
        
        _, self.dim = self.X_0.shape
        
        create_net = self.create_network if  is None else create_network
        
        self.b_forward_init, self.b_forward = create_net(
            self.dim, weights
        )
        self.b_backward_init, self.b_backward = create_net(
            self.dim, weights
        )
        
        self.sigma = lambda X,t: sigma_sq
        
        self.rng = rng
        
        self.opt_init_f, self.opt_update_f, self.get_params_f = (
            optimizers.momentum(step_size, mass=momentum_mass)
        )
        
        self.opt_init_b, self.opt_update_b, self.get_params_b = (
            optimizers.momentum(step_size, mass=momentum_mass)
        )
        
        num_complete_batches_f, leftover_f = divmod(self.X_0.shape[0], self.batch_size_f)
        self.num_batches_f = num_complete_batches_f + bool(leftover_f)
                                               
        num_complete_batches_b, leftover_b = divmod(self.X_1.shape[0], self.batch_size_b)
        self.num_batches_b = num_complete_batches_b + bool(leftover_b)
        
    
    @staticmethod
    def create_network(dim, weights):
        
        model  = []
        for weight in weights:
            model.append(
                Dense(weight)
            )
            
            model.append(
                Relu(weight)
            )
            
        
        model.append(dim)
        init_random_params, predict = stax.serial(
            *model
        )
        return init_random_params, predict
        
    @staticmethod
    def loss_for_trajectory(Xt, b_f, b_b, dt, theta, forwards = True):
        b_minus  = b_b(Xt)
        b_plus = b_f(Xt)
        
        delta_Xt = Xt[:-1, :]  - Xt[1:, :]
        
        sign = 1.0 if forwards else -1.0
        
        ito_integral = sign *  (b_plus[1:,:] - b_minus[:-1,:])  * delta_Xt
        
        time_integral = sign *  (b_plus**2 - b_minus**2) * dt
        
        return ito_integral.sum() - 0.5 * time_integral.sum()
        
    def data_stream(self, forward=True):
        rng = self.rng
        X = self.X_0 if forward else self.X_1
        
        batch_size = batch_size_f if forward else batch_size_b
        num_batches = self.num_batches_f if forward else self.num_batches_b
        
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield X[batch_idx] 

    @jit
    def update(self, i, opt_state, batch, forwards=True):
        params = self.get_params(opt_state)
        
        loss = lambda params, batch: self.inner_loss(params, batch, forwards=forwards)
        return self.opt_update(i, grad(loss)(self.theta, batch), opt_state)
        
    def sample_trajectory(self, X, forwards=True):
        
        # backwards discretisation has a sign flip         
        b = self.b_forward if forward else (lambda X, t, theta: -self.b_backward(X, t, theta))
        
        return self.sde_solver(alfa=b, beta=self.sigma,
                               dt=self.dt, X0=X,
                               N=self.number_time_steps)
    
    def inner_loss(self, theta, batch, forwards=True):
                       
        J = 0
        
        for x in batch:
            t, Xt = self.sample_trajectory(x, forwards=forwards)
            
            J += loss_for_trajectory(Xt, self.b_forward, self.b_backward, dt, forwards=forwards, theta)
        
        J /= len(X)
        
        return J

    def fit(self, IPFP_iterations=10, sub_iterations=10):     
        
        _, init_params_f = self.b_forward_init(self.rng, (-1, self.dim))                                             
        opt_state_f = self.opt_init_f(init_params_f)
        
        _, init_params_b = self.b_backward_init(self.rng, (-1, self.dim))                                               
        opt_state_b = self.opt_init_b(init_params_b)
        
        batches_f = self.data_stream(forwards=True)
        batches_b = self.data_stream(forwards=False)
        
        for i in range(IPFP_iterations):
                                               
            itercount = itertools.count()
            
            for k in range(sub_iterations):
                for _ in range(num_batches_b):
                    
                    opt_state_b = self.update(
                        next(itercount), opt_state_b, next(batches_b), forwards=False
                    )
                                               
            
            itercount = itertools.count()
            
            for k in range(sub_iterations):
                for _ in range(num_batches_f):
                    
                    opt_state_f = self.update(
                        next(itercount), opt_state_f, next(batches_f), forwards=True
                    )
         

        self.theta_f = get_params(opt_state_f)
        self.theta_b = get_params(opt_state_b)
            
        