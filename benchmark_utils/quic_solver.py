from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit


class QUIC():
    def __init__(self,
                 max_iter=100):

        self.max_iter = max_iter

    pass

    def fit(self, S):

        p = S.shape[0]
        W = S.copy()
        W *= 0.95
        diagonal = S.flat[:: p + 1]
        W.flat[:: p + 1] = diagonal
        Theta = np.linalg.pinv(W)

        for it in self.max_iter:

            # 1
            W = np.linalg.pinv(Theta)
