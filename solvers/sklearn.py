from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from sklearn.covariance import GraphicalLasso


class Solver(BaseSolver):

    name = 'sklearn'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha):
        self.S = S
        self.alpha = alpha

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18
        self.model = GraphicalLasso(alpha=self.alpha,
                                    covariance="precomputed",
                                    tol=self.tol,
                                    )

        self.run(2)

    def run(self, n_iter):

        self.model.max_iter = n_iter
        self.model.fit(self.S)

        self.Theta = self.model.precision_

    def get_result(self):
        return dict(Theta=self.Theta)
