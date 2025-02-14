from benchopt import BaseDataset, safe_import_context

from sklearn.utils import check_random_state
from sklearn.datasets import make_sparse_spd_matrix

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "simulated"

    parameters = {
        'n_samples, n_features': [
            (1000, 100),
            (1000, 250),
            (1000, 400),
        ],
        'sparsity_controller': [0.9],
        'random_state': [0],
    }

    requirements = ["sklearn"]

    def get_data(self):
        rng = check_random_state(self.random_state)
        Theta_true = make_sparse_spd_matrix(
            self.n_features,
            alpha=self.sparsity_controller,
            random_state=rng)

        Theta_true += 0.1*np.eye(self.n_features)
        Sigma_true = np.linalg.pinv(Theta_true, hermitian=True)
        X = rng.multivariate_normal(
            mean=np.zeros(self.n_features),
            cov=Sigma_true,
            size=self.n_samples,
        )
        S = np.cov(X, bias=True, rowvar=False)

        return dict(S=S,
                    Theta_true=Theta_true,
                    )
