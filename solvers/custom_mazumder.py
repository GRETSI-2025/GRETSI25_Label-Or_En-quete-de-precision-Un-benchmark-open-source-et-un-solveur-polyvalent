from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils import GraphicalLasso


class Solver(BaseSolver):

    name = 'Custom'

    parameters = {
        # 'scale_step': [1, 1.99],  # ?
    }

    requirements = ["numpy"]

    def set_objective(self, Theta, S, alpha):
        self.Theta = Theta
        self.S = S
        self.alpha = alpha

        self.model = GraphicalLasso(alpha=self.alpha,
                                    algo="mazumder",
                                    warm_start=True,
                                    tol=1e-4)

    def run(self, n_iter):

        self.model.max_iter = n_iter
        self.model.fit(self.S)

        self.Theta = self.model.precision_

    def get_result(self):
        return dict(Theta=self.Theta)
