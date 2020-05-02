import jax.numpy as np



def fast_mahalanobis_2(X, Y , Lambda, partition=True):
    _, dim = X.shape

    s, logdet = np.linalg.slogdet(Lambda)
    # import  pdb; pdb.set_trace()
    xt = np.dot(X, Lambda)
    yt = np.dot(Y, Lambda)

    cross_dot = np.dot(xt, Y.t())

    xdot = ( xt * X ).sum(dim=1)
    ydot = ( yt * Y).sum(dim=1)

    exponent = -((xdot[..., None] - 2.0 * cross_dot) + ydot[None, ...])
    neg_log_Z = 0.5 * ( logdet + dim * np.log(np.pi) )
    
    fac = 1.0 if partition else 0.0
    # import pdb; pdb.set_trace()
    return exponent + fac * neg_log_Z