import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    # from gglasso.problem import glasso_problem
    from gglasso.solver.single_admm_solver import ADMM_SGL


class Solver(BaseSolver):
    name = 'gglasso'

    parameters = {
        "inner_tol": [1e-4],
    }

    requirements = ["numpy"]

    def set_objective(self, S, alpha):
        self.S = S
        self.alpha = alpha

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18
        # self.model = glasso_problem(
        #     S,
        #     N=1000,
        #     reg='SGL',
        #     reg_params={'lambda1': self.alpha},
        #     latent=False,
        #     do_scaling=False)

        # Same as for skglm
        # self.run(5)

    def run(self, n_iter):

        # self.model.solve(solver_params={
        #     # 'tol': self.tol,
        #     # 'max_iter': n_iter
        # })

        if n_iter > 0:
            sol = ADMM_SGL(S=self.S,
                           lambda1=self.alpha,
                           # This is what their wrapper does
                           Omega_0=np.eye(self.S.shape[0]),
                           max_iter=n_iter,
                           tol=self.tol
                           )
            self.Theta = sol[0]['Theta']
        else:
            self.Theta = np.eye(self.S.shape[0])

        # self.Theta = self.model.solution.adjacency_

    def get_result(self):
        return dict(Theta=self.Theta)
