import numpy as np

def project_box(x, lb, ub):
    if lb is not None: x = np.maximum(x, lb)
    if ub is not None: x = np.minimum(x, ub)
    return x

def grp_soft_threshold(x, groups, thresh):
    """
    Exact output for non-overlapping groups
    One pass for overlapping ones in Gauss-Siedel fashion
    """
    x_new = x.copy()
    for G in groups:
        v = x_new[G]
        nrm = np.linalg.norm(v)
        if nrm > thresh:
            x_new[G] = (1.0 - thresh/nrm) * v
        else:
            x_new[G] = 0.0
    return x_new


class PDHGStatus:
    OPTIMAL: 0
    MAX_ITER: 1
    INFEASIBLE: 2


class GroupPDHG:
    """
    Solve:
        min_x c^T x + mu * sum_i ||x_{G_i}||_2
        s.t. A x <= b
                P x = r
                lb_x <= x <= ub_x
    """
    def __init__(
            self,
            c, groups, mu,
            lb_x=None, ub_x=None,
            tau=1e-2, sigma=1e-2, theta=1.0, max_iter=5000, 
            tol=1e-6, tol_cons=1e-6, dual_max=1e6, 
            verbose=False
    ):
        self.c = np.asarray(c)
        self.groups = groups
        self.mu = mu
        self.lb_x = lb_x
        self.ub_x = ub_x
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.max_iter = max_iter
        self.tol = tol
        self.tol_cons = tol_cons
        self.dual_max = dual_max
        self.verbose = verbose

    def solve(self, A, b, P=None, r=None, x0=None, lam0=None, nu0=None):
        n = self.c.size
        m = A.shape[0]
        if P is None:
            P = np.zeros((0, n))
            r = np.zeros(n)
        p = P.shape
        # initialization
        if x0 is None: x = np.zeros(n)
        else: x = x0.copy()
        x_bar = x.copy()
        lam = np.zeros(m) if lam0 is None else lam0.copy()
        nu = np.zeros(p) if nu0 is None else nu0.copy()
        # pdhg loop
        status = PDHGStatus.MAX_ITER
        for k in range(self.max_iter):
            # dual update
            lam_old = lam.copy()
            nu_old = nu.copy()
            lam += self.sigma * (A @ x_bar - b)
            lam = np.maximum(lam, 0.0)
            if p > 0: nu += self.sigma * (P @ x_bar - r)
            # dual test
            dual_norm = max(
                np.linalg.norm(lam, np.inf),
                np.linalg.norm(nu, np.inf) if p > 0 else 0.0,
            )
            if dual_norm > self.dual_max:
                if self.verbose:
                    print(f"PDHG infeasible: dual blow-up at iter {k}")
                status = PDHGStatus.INFEASIBLE
                break
            # primal update
            g = self.c + A.T @ lam + (P.T @ nu if p > 0 else 0.0)
            v = x - self.tau * g
            x_new = grp_soft_threshold(v, self.groups, self.tau * self.mu)
            x_new = project_box(x_new, self.lb_x, self.ub_x)
            x_bar = x_new + self.theta * (x_new - x)
            # convergence
            dx = np.linalg.norm(x_new - x)
            dlam = np.linalg.norm(lam - lam_old)
            dnu = np.linalg.norm(nu - nu_old)
            x = x_new
            if max(dx, dlam, dnu) < self.tol:
                status = PDHGStatus.OPTIMAL
                if self.verbose:
                    print(f"PDHG converged in {k} iterations")
                break
        # feasibility checks
        if status == PDHGStatus.MAX_ITER:
            ineq_violation = np.linalg.norm(np.maximum(A @ x - b, 0.0), np.inf)
            eq_violation = (
                np.linalg.norm(P @ x - r, np.inf) if p > 0 else 0.0
            )
            if max(ineq_violation, eq_violation) > self.tol_cons:
                status = PDHGStatus.INFEASIBLE
                if self.verbose:
                    print("PDHG infeasible: constraint violation after max_iter")
        fval = (
            self.c @ x
            + self.mu * sum(np.linalg.norm(x[G]) for G in self.groups)
        )
        return status, fval, x, lam, nu
