from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):

    name = "GLasso objective"

    url = "https://github.com/Perceptronium/benchmark_graphical_lasso"

    alphas = self.alpha_max*np.geomspace(1, 1e-3, num=30)

    # alpha is the regularization hyperparameter
    parameters = {
        'alpha': alphas,
    }

    requirements = ["numpy"]

    min_benchopt_version = "1.5"

    def set_data(self, S, Theta_true, alpha_max):

        self.S = S
        self.Theta_true = Theta_true
        self.alpha_max = alpha_max

    def evaluate_result(self, Theta):

        neg_llh = (-np.slogdet(Theta)[1] +
                   np.trace(Theta @ self.S))  # The trace can be computed smarter ?

        pen = self.alpha*np.sum(np.abs(Theta))

        return dict(
            value=neg_llh + pen,
            neg_log_likelihood=neg_llh,
            penalty=pen
        )

    def get_one_result(self):
        return dict(Theta=np.zeros_like(self.S))

    def get_objective(self):
        return dict(
            Theta=self.Theta,
            S=self.S,
            alpha=self.alpha
        )
