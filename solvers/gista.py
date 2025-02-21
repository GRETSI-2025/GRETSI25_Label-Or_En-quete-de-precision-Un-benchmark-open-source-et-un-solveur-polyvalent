import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import GraphicalIsta


class Solver(BaseSolver):
    name = 'g-ista'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.S = S
        self.alpha = alpha

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18
        self.model = GraphicalIsta(alpha=self.alpha,
                                   tol=self.tol,
                                   )
        # Same as for skglm
        self.run(3)

    def run(self, n_iter):

        self.model.max_iter = n_iter
        self.model.fit(self.S)

        self.Theta = self.model.precision_

    def get_result(self):
        return dict(Theta=self.Theta)
