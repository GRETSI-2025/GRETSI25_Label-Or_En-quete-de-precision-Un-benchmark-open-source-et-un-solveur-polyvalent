from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from sklearn.utils.validation import check_random_state
    from sklearn.linear_model import _cd_fast as cd_fast

    import scipy

    from skglm.solvers import AndersonCD
    from skglm.datafits import QuadraticHessian
    from skglm.penalties import WeightedL1
    from skglm.utils.jit_compilation import compiled_clone


class GraphicalLasso():
    def __init__(self,
                 alpha=1.,
                 weights=None,
                 algo="banerjee",
                 max_iter=1000,
                 tol=1e-8,
                 warm_start=False,
                 ):
        self.alpha = alpha
        self.weights = weights
        self.algo = algo
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, S):
        p = S.shape[-1]
        indices = np.arange(p)

        if self.weights is None:
            Weights = np.ones((p, p))
        else:
            Weights = self.weights
            if not np.allclose(Weights, Weights.T):
                raise ValueError("Weights should be symmetric.")

        if self.warm_start and hasattr(self, "precision_"):
            if self.algo == "banerjee":
                raise ValueError(
                    "Banerjee does not support warm start for now.")
            Theta = self.precision_
            W = self.covariance_
        else:
            # W = S.copy()  # + alpha*np.eye(p)
            W = S.copy()
            W *= 0.95
            W.flat[:: p + 1] = S.flat[:: p + 1]
            # Theta = np.linalg.pinv(W, hermitian=True)
            Theta = scipy.linalg.pinvh(W)

        # datafit = compiled_clone(QuadraticHessian())

        # penalty = compiled_clone(
        #     WeightedL1(alpha=self.alpha, weights=Weights[0, :-1]))

        # solver = AndersonCD(warm_start=True,
        #                     fit_intercept=False,
        #                     ws_strategy="fixpoint")

        for it in range(self.max_iter):
            Theta_old = Theta.copy()
            for col in range(p):
                indices_minus_col = np.concatenate(
                    [indices[:col], indices[col + 1:]])
                _11 = indices_minus_col[:, None], indices_minus_col[None]
                _12 = indices_minus_col, col
                _21 = col, indices_minus_col
                _22 = col, col

                W_11 = W[_11]
                w_12 = W[_12]
                w_22 = W[_22]
                s_12 = S[_12]
                s_22 = S[_22]

                # penalty.weights = Weights[_12]

                # if self.algo == "banerjee":
                #     w_init = Theta[_12]/Theta[_22]
                #     Xw_init = W_11 @ w_init
                #     Q = W_11
                # elif self.algo == "mazumder":
                #     inv_Theta_11 = W_11 - np.outer(w_12, w_12)/w_22
                #     Q = inv_Theta_11
                #     w_init = Theta[_12] * w_22
                #     Xw_init = inv_Theta_11 @ w_init
                # else:
                #     raise ValueError(f"Unsupported algo {self.algo}")

                # beta, _, _ = solver._solve(
                #     Q,
                #     s_12,
                #     datafit,
                #     penalty,
                #     w_init=w_init,
                #     Xw_init=Xw_init,
                # )

                enet_tol = 1e-4  # same as sklearn
                eps = np.finfo(np.float64).eps
                beta = -(Theta[_12] / (Theta[_22] + 1000*eps))
                beta, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                    beta,
                    self.alpha,
                    0,
                    W_11,
                    s_12,
                    s_12,
                    self.max_iter,
                    enet_tol,
                    check_random_state(None),
                    False,
                )

                if self.algo == "banerjee":
                    # w_12 = -W_11 @ beta
                    w_12 = W_11 @ beta

                    W[_12] = w_12
                    W[_21] = w_12

                    # Theta[_22] = 1/(s_22 + beta @ w_12)
                    Theta[_22] = 1/(w_22 - beta @ w_12)

                    # Theta[_12] = beta*Theta[_22]
                    Theta[_12] = -beta*Theta[_22]

                # else:  # mazumder
                #     theta_12 = beta / s_22
                #     theta_22 = 1/s_22 + theta_12 @ inv_Theta_11 @ theta_12

                #     Theta[_12] = theta_12
                #     Theta[_21] = theta_12
                #     Theta[_22] = theta_22

                #     w_22 = 1/(theta_22 - theta_12 @ inv_Theta_11 @ theta_12)
                #     w_12 = -w_22*inv_Theta_11 @ theta_12
                #     W_11 = inv_Theta_11 + np.outer(w_12, w_12)/w_22
                #     W[_11] = W_11
                #     W[_12] = w_12
                #     W[_21] = w_12
                #     W[_22] = w_22

            if norm(Theta - Theta_old) < self.tol:
                print(f"Weighted Glasso converged at CD epoch {it + 1}")
                break
        else:
            print(
                # f"Not converged at epoch {it + 1}, "
                # f"diff={norm(Theta - Theta_old):.2e}"
            )
        self.precision_, self.covariance_ = Theta, W
        # self.n_iter_ = it + 1

        return self
