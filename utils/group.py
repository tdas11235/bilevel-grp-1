import cvxpy as cp
import numpy as np
from enum import Enum

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


def estimate_operator_norm(A, n_iter=30):
    """
    Estimate ||A||_2 using power iteration.
    Returns spectral norm estimate.
    """
    n = A.shape[1]
    x = np.random.randn(n)
    x /= np.linalg.norm(x)
    for _ in range(n_iter):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny < 1e-12:
            return 0.0
        x = A.T @ y
        nx = np.linalg.norm(x)
        if nx < 1e-12:
            return 0.0
        x /= nx
    return np.linalg.norm(A @ x)


class InnerStatus(Enum):
    OPTIMAL = 0
    MAX_ITER = 1
    INFEASIBLE = 2


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
            verbose=True
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
            r = np.zeros(0)
        p = P.shape[0]
        # initialization
        if x0 is None: x = np.zeros(n)
        else: x = x0.copy()
        x_bar = x.copy()
        lam = np.zeros(m) if lam0 is None else lam0.copy()
        nu = np.zeros(p) if nu0 is None else nu0.copy()
        # pdhg loop
        status = InnerStatus.MAX_ITER
        for k in range(self.max_iter):
            # dual update
            lam_old = lam.copy()
            nu_old = nu.copy()
            # L = estimate_operator_norm(A)  # or power iteration approx
            # tau = 0.9 / L
            # sigma = 0.9 / L
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
                status = InnerStatus.INFEASIBLE
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
            fval = (
                self.c @ x
                + self.mu * sum(np.linalg.norm(x[G]) for G in self.groups)
            )
            # print(fval, dx, dlam, dnu)
            ineq_violation = np.linalg.norm(np.maximum(A @ x - b, 0.0), np.inf)
            eq_violation = (
                np.linalg.norm(P @ x - r, np.inf) if p > 0 else 0.0
            )
            if max(dx, dlam, dnu) < self.tol and max(ineq_violation, eq_violation) <= self.tol_cons:
                status = InnerStatus.OPTIMAL
                if self.verbose:
                    print(f"PDHG converged in {k} iterations")
                break
        # feasibility checks
        if status == InnerStatus.MAX_ITER:
            ineq_violation = np.linalg.norm(np.maximum(A @ x - b, 0.0), np.inf)
            eq_violation = (
                np.linalg.norm(P @ x - r, np.inf) if p > 0 else 0.0
            )
            if max(ineq_violation, eq_violation) > self.tol_cons:
                status = InnerStatus.INFEASIBLE
                if self.verbose:
                    print(max(ineq_violation, eq_violation))
                    print("PDHG infeasible: constraint violation after max_iter")
        fval = (
            self.c @ x
            + self.mu * sum(np.linalg.norm(x[G]) for G in self.groups)
        )
        print(status)
        return status, fval, x, lam, nu


class GroupSOCP:
    """
    Solve

        min_x  c^T x + mu * sum_i ||x_{G_i}||_2

        s.t.
                A x <= b          (dual: lambda)
                P x  = r          (dual: nu)
                lb_x <= x <= ub_x

    using CVXPY + ECOS (SOCP interior-point solver).

    Returns:
        status
        fval
        x
        lambda   (for A x <= b)
        nu       (for P x = r)

    Notes
    -----
    - lambda corresponds ONLY to A x <= b
    - bound duals are not returned
    - if infeasible, x/lambda/nu may be None
    """

    def __init__(
        self,
        c,
        groups,
        mu,
        lb_x=None,
        ub_x=None,
        verbose=True,
        abstol=1e-8,
        reltol=1e-8,
        feastol=1e-8,
        max_iters=500,
    ):
        self.c = np.asarray(c, dtype=float)
        self.groups = groups
        self.mu = float(mu)

        self.lb_x = None if lb_x is None else np.asarray(lb_x, dtype=float)
        self.ub_x = None if ub_x is None else np.asarray(ub_x, dtype=float)

        self.verbose = verbose

        self.abstol = abstol
        self.reltol = reltol
        self.feastol = feastol
        self.max_iters = max_iters

    def solve(
        self,
        A,
        b,
        P=None,
        r=None,
        x0=None,
        lam0=None,
        nu0=None,
    ):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        n = self.c.size
        m = A.shape[0]

        if P is None:
            P = np.zeros((0, n))
            r = np.zeros(0)

        P = np.asarray(P, dtype=float)
        r = np.asarray(r, dtype=float)
        p = P.shape[0]
        # primal variable
        x = cp.Variable(n)

        # group regularization
        group_penalty = 0
        for G in self.groups:
            group_penalty += cp.norm(x[G], 2)

        objective = cp.Minimize(
            self.c @ x + self.mu * group_penalty
        )

        constraints = []
        # main inequality constraints
        ineq_constraint = (A @ x <= b)
        constraints.append(ineq_constraint)
        # equality constraints
        if p > 0:
            eq_constraint = (P @ x == r)
            constraints.append(eq_constraint)
        else:
            eq_constraint = None
        # lower bounds
        if self.lb_x is not None:
            constraints.append(x >= self.lb_x)
        # upper bounds
        if self.ub_x is not None:
            constraints.append(x <= self.ub_x)

        problem = cp.Problem(objective, constraints)

        # optional warm start
        if x0 is not None:
            try:
                x.value = np.asarray(x0, dtype=float)
            except Exception:
                pass
        try:
            problem.solve(
                solver=cp.ECOS,
                verbose=self.verbose,
                warm_start=True,
                abstol=self.abstol,
                reltol=self.reltol,
                feastol=self.feastol,
                max_iters=self.max_iters,
            )
        except Exception as e:
            if self.verbose:
                print(f"ECOS solver error: {str(e)}")
            return (
                "solver_error",
                None,
                None,
                None,
                None,
            )
        status = problem.status
        if status not in ["optimal", "optimal_inaccurate"]:
            if self.verbose:
                print(f"ECOS status: {status}")
            return (
                InnerStatus.INFEASIBLE,
                None,
                None,
                None,
                None,
            )

        x_star = np.asarray(x.value).reshape(-1)
        # duals ONLY for A x <= b
        lam = np.asarray(ineq_constraint.dual_value).reshape(m)
        # duals for equality constraints
        if p > 0:
            nu = np.asarray(eq_constraint.dual_value).reshape(p)
        else:
            nu = np.zeros(0)
        fval = (
            self.c @ x_star
            + self.mu * sum(
                np.linalg.norm(x_star[G]) for G in self.groups
            )
        )
        if self.verbose:
            print("ECOS converged successfully")
        return (
            InnerStatus.OPTIMAL,
            fval,
            x_star,
            lam,
            nu,
        )
