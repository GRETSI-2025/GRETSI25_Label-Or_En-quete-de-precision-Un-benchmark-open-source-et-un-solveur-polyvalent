from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit


@njit
def neg_llh(Theta, S):
    return (-np.linalg.slogdet(Theta)[1] + (Theta * S).sum())


@njit
def loss(Theta, S, alpha):
    """ Compute Graphical Lasso loss function"""

    neg_llh = neg_llh(Theta, S)
    pen = alpha * np.sum(np.abs(Theta))
    # diagonal is not penalized:
    pen -= alpha * np.trace(np.abs(Theta))
    return neg_llh + pen
