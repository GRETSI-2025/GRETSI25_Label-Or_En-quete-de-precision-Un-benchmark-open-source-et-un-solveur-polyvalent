from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from sklearn.utils.validation import check_random_state
    from sklearn.linear_model import _cd_fast as cd_fast

    from numba import njit
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
                 lasso_solver="anderson_cd",
                 max_iter=1000,
                 tol=1e-8,
                 warm_start=False,
                 inner_tol=1e-4,
                 ):
        self.alpha = alpha
        self.weights = weights
        self.algo = algo
        self.lasso_solver = lasso_solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.inner_tol = inner_tol

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
            W = S.copy()
            W *= 0.95
            W.flat[:: p + 1] = S.flat[:: p + 1]
            # Theta = np.linalg.pinv(W, hermitian=True)
            Theta = scipy.linalg.pinvh(W)

        datafit = compiled_clone(QuadraticHessian())

        penalty = compiled_clone(
            WeightedL1(alpha=self.alpha, weights=Weights[0, :-1]))

        solver = AndersonCD(
            warm_start=True,
            fit_intercept=False,
            ws_strategy="subdiff",
            tol=self.inner_tol,
        )

        W_11 = np.copy(W[1:, 1:], order="C")
        for it in range(self.max_iter):
            Theta_old = Theta.copy()
            for col in range(p):
                if col > 0:
                    di = col - 1
                    W_11[di] = W[di][indices != col]
                    W_11[:, di] = W[:, di][indices != col]
                else:
                    W_11[:] = W[1:, 1:]

                s_12 = S[col, indices != col]

                if self.lasso_solver == "anderson_cd":
                    # penalty.weights = Weights[_12]
                    if self.algo == "banerjee":
                        eps = np.finfo(np.float64).eps
                        w_init = (Theta[indices != col, col] /
                                  (Theta[col, col] + 1000 * eps))
                        Xw_init = W_11 @ w_init
                        Q = W_11
                    # elif self.algo == "mazumder":
                    #     inv_Theta_11 = W_11 - np.outer(w_12, w_12)/w_22
                    #     Q = inv_Theta_11
                    #     w_init = Theta[_12] * w_22
                    #     Xw_init = inv_Theta_11 @ w_init
                    # else:
                    #     raise ValueError(f"Unsupported algo {self.algo}")

                    beta, _, _ = solver._solve(
                        Q,
                        s_12,
                        datafit,
                        penalty,
                        w_init=w_init,
                        Xw_init=Xw_init,
                    )
                elif self.lasso_solver == "cd_fast":
                    enet_tol = 1e-4  # same as sklearn
                    eps = np.finfo(np.float64).eps
                    beta = -(Theta[indices != col, col]
                             / (Theta[col, col] + 1000 * eps))
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
                    beta = -beta
                elif self.lasso_solver == "cd_numba":
                    beta = (Theta[indices != col, col] /
                            (Theta[col, col] + 1e-13))
                    beta = cd_gram(
                        W_11,
                        s_12,
                        x=beta,
                        alpha=self.alpha,
                        tol=self.inner_tol,
                    )

                if self.algo == "banerjee":
                    w_12 = -W_11 @ beta  # for us
                    # w_12 = W_11 @ beta

                    W[col, indices != col] = w_12
                    W[indices != col, col] = w_12

                    Theta[col, col] = 1/(W[col, col] + beta @ w_12)  # For us
                    # Theta[col, col] = 1/(W[col, col] - beta @ w_12)

                    Theta[indices != col, col] = beta*Theta[col, col]  # For us
                    Theta[col, indices != col] = beta*Theta[col, col]
                    # Theta[indices != col, col] = -beta*Theta[col, col]
                    # Theta[col, indices != col] = -beta*Theta[col, col]

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


@njit
def ST(x, tau):
    if x > tau:
        return x-tau
    elif x < -tau:
        return x + tau
    else:
        return 0


@njit
def cd_gram(H, q, x, alpha, max_iter=1000, tol=1e-4):
    """
    Solve min .5 * x.T H x + q.T @ x + alpha * norm(x, 1).

    H must be symmetric.
    """
    dim = H.shape[0]
    lc = np.zeros(dim)
    for j in range(dim):
        lc[j] = H[j, j]

    Hx = H @ x
    for epoch in range(max_iter):
        max_delta = 0  # max coeff change

        for j in range(dim):

            x_j_prev = x[j]
            x[j] = ST(x[j] - (Hx[j] + q[j]) / lc[j], alpha/lc[j])
            max_delta = max(max_delta, np.abs(x_j_prev - x[j]))

            if x_j_prev != x[j]:
                Hx += (x[j] - x_j_prev) * H[j]

        # print(epoch, max_delta)
        if max_delta < tol:
            break

    return x


@njit
def anderson_cd_gram(H, q, x, alpha, max_iter=1000, tol=1e-4):
    """
    Solve cd_gram with extrapolation.

    H must be symmetric.
    """

    anderson_mem_size = 4
    anderson_mem = np.zeros((x.shape[0], anderson_mem_size+1))

    dim = H.shape[0]
    lc = np.zeros(dim)
    for j in range(dim):
        lc[j] = H[j, j]

    Hx = H @ x
    for epoch in range(max_iter):
        max_delta = 0  # max coeff change

        for j in range(dim):

            x_j_prev = x[j]
            x[j] = ST(x[j] - (Hx[j] + q[j]) / lc[j], alpha/lc[j])
            max_delta = max(max_delta, np.abs(x_j_prev - x[j]))

            if x_j_prev != x[j]:
                Hx += (x[j] - x_j_prev) * H[j]

        # print(epoch, max_delta)
        if max_delta < tol:
            break

        anderson_mem[:, epoch]

        if epoch > anderson_mem_size:
            U = np.diff(anderson_mem, axis=1)

            c = np.solve(U.T @ U, np.ones(anderson_mem_size))

            x = c / np.sum(c)

    return x
