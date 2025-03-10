from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    # You need to clone and install https://github.com/skggm/skggm
    from inverse_covariance import QuicGraphicalLasso


class Solver(BaseSolver):
    name = 'skggm'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.S = S
        self.alpha = alpha
        self.X = X

        # W = S.copy()
        # W *= 0.95
        # diagonal = S.flat[:: S.shape[0] + 1]
        # W.flat[:: S.shape[0] + 1] = diagonal
        # Theta_init = np.linalg.pinv(W, hermitian=True)

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18
        self.model = QuicGraphicalLasso(lam=self.alpha,
                                        mode="default",
                                        auto_scale=False,
                                        init_method="cov",
                                        # Theta0=Theta_init,
                                        # Sigma0=W,
                                        tol=self.tol,
                                        )
        # Same as for skglm
        # self.run(5)

    def run(self, n_iter):

        self.model.max_iter = n_iter
        self.model.fit(self.X)

        self.Theta = self.model.precision_

    def get_result(self):
        return dict(Theta=self.Theta)
