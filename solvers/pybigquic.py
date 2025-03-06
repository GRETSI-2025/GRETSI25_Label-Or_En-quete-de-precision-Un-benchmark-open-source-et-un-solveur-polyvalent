import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.py_bigquic.py_bigquic import bigquic


class Solver(BaseSolver):
    name = 'pybigquic'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.X = X
        self.alpha = alpha

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18

        self.run(2)

    def run(self, n_iter):

        self.Theta = bigquic(data=self.X, alpha=self.alpha,
                             max_iter=n_iter, tol=self.tol)

    def get_result(self):
        return dict(Theta=self.Theta)
