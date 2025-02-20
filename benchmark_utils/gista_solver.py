from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit


class GraphicalIsta():
    def __init__(self,
                 alpha=1.,
                 gamma=1.,
                 max_iter=100,
                 tol=1e-8,
                 warm_start=False):

        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, S):

        Theta, W = gista_fit(S, self.alpha, self.gamma, self.max_iter)

        self.precision_, self.covariance_ = Theta, W
        return self


@njit
def ST_off_diag(x, tau):
    off_diag = np.sign(x) * np.maximum(np.abs(x) - tau, 0)
    diag = np.diag(x)
    np.fill_diagonal(off_diag, diag)
    return off_diag


@njit
def gista_fit(S, alpha, gamma, max_iter):

    p = S.shape[0]
    W = S.copy()
    W *= 0.95
    # diagonal = S.flat[:: p + 1]
    diagonal = np.diag(S)
    np.fill_diagonal(W, diagonal)
    Theta = np.linalg.pinv(W)

    for iter in range(max_iter):
        Theta = ST_off_diag(Theta - gamma*(S - W), alpha*gamma)
        W = np.linalg.pinv(Theta)

    return Theta, W
