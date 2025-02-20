from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np

    from sklearn.exceptions import ConvergenceWarning

    from benchmark_utils import GraphicalLasso


class Solver(BaseSolver):

    name = 'skglm'

    parameters = {
        'algo': [
            "banerjee",
            # "mazumder",
        ],
        'lasso_solver': [
            "cd_fast",
            "cd_numba",
            "anderson_cd_numba",
        ],
        'outer_anderson': [
            True,
            False,
        ]}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.S = S
        self.alpha = alpha

        # to stay comparable to sklearn solver
        self.tol = 1e-18
        self.model = GraphicalLasso(alpha=self.alpha,
                                    algo=self.algo,
                                    lasso_solver=self.lasso_solver,
                                    warm_start=False,
                                    outer_anderson=self.outer_anderson,
                                    tol=self.tol,
                                    inner_tol=1e-4,
                                    )
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        # Cache Numba compilation
        self.run(2)

    def run(self, n_iter):

        self.model.max_iter = n_iter
        self.model.fit(self.S)

        self.Theta = self.model.precision_

    def get_result(self):
        return dict(Theta=self.Theta)
