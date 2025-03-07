import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    # from benchmark_utils.py_bigquic.py_bigquic import bigquic
    from sklearn.covariance import GraphicalLasso
    try:
        import rpy2
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        R = rpy2.robjects.r
        importr('BigQuic', lib_loc='/home/cpouliquen/R/x86_64-pc-linux-gnu-library/4.4')
    except:
        msg = "Either R, rpy2, or BigQuic are not installed"
        raise ImportError(msg)


class Solver(BaseSolver):
    name = 'pybigquic'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.X = X
        self.alpha = alpha
        self.ralpha = numpy2ri.py2rpy(self.alpha)
        self.S = S

        self.tol = 1e-18
        self.rtol = numpy2ri.py2rpy(self.tol)

        data = np.copy(self.X)
        # data -= data.mean(axis=0)
        self.stds = np.std(data, axis=0)
        data /= self.stds

        program_string = f"""
        bigquic <- function(data,alpha,max_iter,tol) {{
        out <- BigQuic(X=data, lambda=alpha, isnormalized=1, use_ram=TRUE, maxit=max_iter, epsilon=tol)
        as(out$precision_matrices[[1]], "matrix")
        }}
        """
        R(program_string)

        self.rdata = numpy2ri.py2rpy(data)
        self.bigquic = R['bigquic']

        self.sk_prec = GraphicalLasso(
            alpha=self.alpha, covariance="precomputed", max_iter=1000, tol=1e-8).fit(self.S).precision_
        self.sk_diag = self.sk_prec.diagonal()

    def run(self, n_iter):

        self.rn_iter = numpy2ri.py2rpy(n_iter)
        self.Theta = np.array(self.bigquic(self.rdata, self.ralpha, self.rn_iter, self.rtol))[
            :self.S.shape[0], :self.S.shape[0]]

        self.Theta = 1. / self.stds * self.Theta / self.stds[:, np.newaxis]

        # np.fill_diagonal(self.Theta, self.sk_diag)

    def get_result(self):
        return dict(Theta=self.Theta)
