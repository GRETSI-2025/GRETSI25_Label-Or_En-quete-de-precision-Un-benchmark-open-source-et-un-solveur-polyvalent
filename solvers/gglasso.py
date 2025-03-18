import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from gglasso.solver.single_admm_solver import ADMM_SGL


class Solver(BaseSolver):
    name = 'gglasso'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.S = S
        self.alpha = alpha

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18

    def run(self, n_iter):

        # gglasso is not robust to max_iter=0
        if n_iter > 0:
            sol = ADMM_SGL(S=self.S,
                           lambda1=self.alpha,
                           Omega_0=np.eye(self.S.shape[0]),
                           max_iter=n_iter,
                           tol=self.tol
                           )
            self.Theta = sol[0]['Theta']
        else:
            self.Theta = self.Theta_init

    def get_result(self):
        return dict(Theta=self.Theta)
