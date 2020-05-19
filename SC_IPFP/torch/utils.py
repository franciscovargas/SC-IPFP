import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.pi = torch.tensor(math.pi).double()

def fast_mahalanobis_2(X, Y , Lambda, partition=True):
    _, dim = X.shape

    s, logdet = torch.slogdet(Lambda)
    # import  pdb; pdb.set_trace()
    xt = X.mm(Lambda)
    yt = Y.mm(Lambda)

    cross_dot = xt.mm(Y.t())

    xdot = ( xt * X ).sum(dim=1)
    ydot = ( yt * Y).sum(dim=1)

    exponent = -((xdot[..., None] - 2.0 * cross_dot) + ydot[None, ...])
    neg_log_Z = 0.5 * ( logdet + dim * torch.log(torch.pi) )
    
    fac = 1.0 if partition else 0.0
    # import pdb; pdb.set_trace()
    return exponent + fac * neg_log_Z


def log_kde_pdf_per_point(x_star, X, S):
#     out = np.zeros((x_star.shape[0],1))
    N = torch.tensor(X.shape[0]).double()
    # Can be vectorized but gram matrix is big
    Lambda = 1.0 / S # torch.inverse(S) # diagonal bandwidth
    
    out =  torch.logsumexp( fast_mahalanobis_2(x_star, X, Lambda), dim=1)
    return (out - torch.log(N))

def kde_pdf_per_point(x_star, X, S):
    return torch.exp(log_kde_pdf_per_point(x_star, X, S))


def silvermans_rule(X):
    """Compute the Silverman factor.
    Returns
    -------
    s : float
        The silverman factor.
    """
    neff, d = X.shape
    sigma = torch.std(X, axis=0)
    
    return (neff*(d+2.0)/4.0)**(-1./(d+4)) * torch.diag(sigma)


class NN(nn.Module):

    def __init__(self, input_dim=1,  weight_dim_list=[20,20,20]):
        super(NN, self).__init__()

        self.weight_dim_list = [input_dim] + weight_dim_list
        self.layers = [None for i in weight_dim_list]

        for i in range(len(weight_dim_list)):
            setattr(
                self,
                "f" + str(i),
                nn.Linear(
                    self.weight_dim_list[i], self.weight_dim_list[i + 1]
                )
            )
            self.layers[i]  = getattr(self, "f" + str(i))

            torch.nn.init.xavier_uniform(self.layers[i].weight)
    
    def get_var(self , other=None):
        return torch.tensor([[0]]).double()

    def forward(self, x, var=None):
        functerino = torch.relu
        x_ = x.clone()
        for i_, layer in enumerate(self.layers[:-1]):
            try:
                x = functerino(layer(x))
            except:
                print(x.shape, i_, x_.shape)
                import pdb; pdb.set_trace()
                raise
        try:
            return (self.layers[-1](x))
        except:
            import pdb; pdb.set_trace()
            raise
