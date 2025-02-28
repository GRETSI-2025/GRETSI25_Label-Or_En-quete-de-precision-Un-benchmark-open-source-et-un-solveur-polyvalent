from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.glasso_solver import GraphicalLasso


class AdaptiveGraphicalLasso():
    def __init__(
        self,
        alpha=1.,
        strategy="log",
        n_reweights=5,
        max_iter=1000,
        tol=1e-8,
        warm_start=False,
        # verbose=False,
    ):
        self.alpha = alpha
        self.strategy = strategy
        self.n_reweights = n_reweights
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, S):
        glasso = GraphicalLasso(
            alpha=self.alpha,
            algo="primal",
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=True)
        Weights = np.ones(S.shape)
        self.n_iter_ = []
        for it in range(self.n_reweights):
            glasso.weights = Weights
            glasso.fit(S)
            Theta = glasso.precision_
            if self.strategy == "log":
                Weights = 1/(np.abs(Theta) + 1e-10)
            elif self.strategy == "sqrt":
                Weights = 1/(2*np.sqrt(np.abs(Theta)) + 1e-10)
            elif self.strategy == "mcp":
                gamma = 3.
                Weights = np.zeros_like(Theta)
                Weights[np.abs(Theta) < gamma*self.alpha] = (self.alpha -
                                                             np.abs(Theta[np.abs(Theta) < gamma*self.alpha])/gamma)
            else:
                raise ValueError(f"Unknown strategy {self.strategy}")

            self.n_iter_.append(glasso.n_iter_)
            # TODO print losses for original problem?
            glasso.covariance_ = np.linalg.pinv(Theta, hermitian=True)
        self.precision_ = glasso.precision_
        self.covariance_ = glasso.covariance_
        return self
