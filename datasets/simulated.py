from benchopt import BaseDataset, safe_import_context

from sklearn.utils import check_random_state
from sklearn.datasets import make_sparse_spd_matrix

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated i.i.d. Gaussian"

    parameters = {
        'n_samples, n_features': [
            # (100, 10),
            # (100, 20),
            (100, 50),
        ],
        'alpha': [0.9],
        'random_state': [0],
    }

    requirements = ["sklearn"]

    def get_data(self):
        rng = check_random_state(self.random_state)
        Theta_true = make_sparse_spd_matrix(
            self.n_features,
            alpha=self.alpha,
            random_state=rng)

        Theta_true += 0.1*np.eye(self.n_features)
        Sigma_true = np.linalg.pinv(Theta_true, hermitian=True)
        X = rng.multivariate_normal(
            mean=np.zeros(self.n_features),
            cov=Sigma_true,
            size=self.n_samples,
        )
        S = np.cov(X, bias=True, rowvar=False)
        S_cpy = np.copy(S)
        np.fill_diagonal(S_cpy, 0.)
        alpha_max = np.max(np.abs(S_cpy))

        return dict(S=S,
                    Theta_true=Theta_true,
                    alpha_max=alpha_max)
