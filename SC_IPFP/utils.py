import jax.numpy as np
import jax.scipy as sc
import jax
from jax import jit


def fast_mahalanobis_2(X, Y , Lambda, partition=True):
    _, dim, *els = X.shape

    s, logdet = np.linalg.slogdet(Lambda)
    # import  pdb; pdb.set_trace()
    xt = np.dot(X, Lambda)
    yt = np.dot(Y, Lambda)

    cross_dot = np.dot(xt, Y.T)

    xdot = ( xt * X ).sum(axis=1)
    ydot = ( yt * Y).sum(axis=1)

    exponent = -((xdot[..., None] - 2.0 * cross_dot) + ydot[None, ...])
    neg_log_Z = 0.5 * ( logdet + dim * np.log(np.pi) )
    
    fac = 1.0 if partition else 0.0
    # import pdb; pdb.set_trace()
    return exponent + fac * neg_log_Z


@jit
def log_kde_pdf_per_point(x_star, X, S):
#     out = np.zeros((x_star.shape[0],1))
    N = X.shape[0]
    # Can be vectorized but gram matrix is big
    Lambda = np.linalg.inv(S)
    
    def inner_loop(x):
        return sc.special.logsumexp(
                fast_mahalanobis_2(X, x[None, ...], Lambda),
                axis=0
         )
#     for i in range(x_star.shape[0]):
#         out = jax.ops.index_update(
#             out,
#             jax.ops.index[i,:],
#             sc.special.logsumexp(
#                 fast_mahalanobis_2(X, x_star[i][None, ...], Lambda),
#                 axis=0
#             )
#         )
    out = jax.vmap(inner_loop)(x_star)
    return (out - np.log(N))

def kde_pdf_per_point(x_star, X, S):
    return np.exp(log_kde_pdf_per_point(x_star, X, S))


def silvermans_rule(X):
    """Compute the Silverman factor.
    Returns
    -------
    s : float
        The silverman factor.
    """
    neff, d = X.shape
    sigma = np.std(X, axis=0)
    
    return np.power(neff*(d+2.0)/4.0, -1./(d+4)) * np.diag(sigma)
