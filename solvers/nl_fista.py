import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.GLASSO_Framework.algos.GLASSO.NL_fista import NL_fista


class Solver(BaseSolver):
    name = 'nl-fista'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.S = S
        self.alpha = alpha

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18
        self.model = NL_fista(
            T=None,
            inner_T=None,
            N=self.S.shape[0],
            lam=self.alpha,
            ls_iter=100,
            step_lim=0)

    def run(self, n_iter):

        self.model.T = n_iter
        self.model.inner_T = n_iter

        if n_iter == 0:
            Theta = np.eye(self.S.shape[0])
        else:
            Theta, _, _, _ = self.model.compute(self.S,
                                                A0=None,
                                                status_f=None,
                                                history=None,
                                                test_check_f=None)

        self.Theta = Theta

    def get_result(self):
        return dict(Theta=self.Theta)
