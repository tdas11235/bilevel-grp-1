import numpy as np
from enum import Enum
from dataclasses import dataclass
from utils.group import PDHGStatus
from utils.act import update_active_groups


class StepStatus(Enum):
    ACCEPTED = 0
    RESTART = 1
    TERMINATE = 2

@dataclass
class TerminatedPoint:
    z: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    nu: np.ndarray
    fval: float

    @classmethod
    def empty(cls):
        return cls(
            z=np.array([np.nan]),
            x=np.array([np.nan]),
            lam=np.array([np.nan]),
            nu=np.array([np.nan]),
            fval=np.nan
        )


class GAPTRSolver:
    def __init__(
            self,
            problem, pdhg_solver, retsoration_solver, groups,
            *, eta=0.1, beta=0.5, tau=1e-4, delta0=1.0,
            eps=1e-6, kappa=0.1, eps_off=1e-6,
            max_iter=1000, t_min=1e-8,
            damp=0.7, amp=1.5, switch_ratio=0.7,
            verbose=False 
    ):
        """
        :param problem: BilevelProblem
        :param pdhg_solver: GroupPDHG
        :param retsoration_solver: RestorationNLP
        """
        self.problem = problem
        self.pdhg = pdhg_solver
        self.restoration = retsoration_solver
        self.groups = groups
        # algorithm params
        self.eta = eta
        self.beta = beta
        self.tau = tau
        self.delta0 = delta0
        self.eps = eps
        self.kappa = kappa
        self.eps_off = eps_off
        self.max_iter = max_iter
        self.t_min = t_min
        self.damp = damp
        self.amp = amp
        self.switch_ratio = switch_ratio
        self.verbose = verbose
        # internal global states
        self.restore_count = 0
        self.tol_mode = False
        self.min_g = np.inf
        self.active_prev = None
        self.x0 = 0.5 * np.ones(self.problem.nx)
        self.sol = TerminatedPoint.empty()
    
    def _project_box(self, z):
        return np.clip(z, self.problem.z_min, self.problem.z_max)
    
    def _effective_gradient(self, z, g):
        g_eff = np.where(
            z <= self.problem.z_min + 1e-6, np.minimum(g, 0.0),
            np.where(z >= self.problem.z_max - 1e-6, np.maximum(g, 0.0), g)
        )
        return g_eff
    
    def _is_stationary(self, g_eff, z, fval):
        check1 = np.linalg.norm(g_eff) < self.eps
        if check1:
            return check1
        check2 = self.tol_mode and np.linalg.norm(
            g_eff) / max(1.0, np.abs(fval), np.linalg.norm(z)) < self.eps
        if check2:
            print("Reached stationary point within acceptable tolerance.")
        return check1 or check2
    
    def step(self):
        """
        Performs one trust region iteration with current z
        Returns StepStatus
        """
        z = self.z
        A = np.asarray(self.problem.eval_A(z))
        b = np.asarray(self.problem.eval_b(z)).flatten()
        P = np.asarray(self.problem.eval_P(z))
        r = np.asarray(self.problem.eval_r(z)).flatten()
        # step 1: Inner problem solve
        status, fval, x, lam, nu = self.pdhg.solve(A, b, P, r, x0=self.x0)
        if status != PDHGStatus.OPTIMAL: raise RuntimeError(f"PDHG solver failed with status {status}")
        self.last_x = x
        self.x0 = x
        self.last_lam = lam
        self.last_nu = nu
        self.last_fval = fval
        # step 2: stationarity check
        g = self.problem.grad_Lz(z, x, lam, nu)
        g_eff = self._effective_gradient(z, g)
        if self._is_stationary(g_eff, z, fval):
            self.sol = TerminatedPoint(
                z=z.copy(),
                x=x.copy(),
                lam=lam.copy(),
                nu=nu.copy(),
                fval=fval
            )
            print("Stationary point reached!")
            return StepStatus.TERMINATE
        # step 3: descent direction
        norm_g = np.linalg.norm(g_eff)
        self.min_g = min(self.min_g, norm_g)
        d = -g_eff / norm_g
        # step 4: Group activation test
        grad_x = self.problem.grad_Lx(z, x, lam, nu)
        active, _ = update_active_groups(
            grad_x, self.groups, self.pdhg.mu, 
            self.active_prev, self.eps_off
        )
        # step-5: Acceptance loop
        t = self.delta
        accepted = False
        while t > self.t_min:
            dz = t * d
            z_trial = self._project_box(z + dz)
            dz_eff = z_trial - z
            # projection killed step
            if np.linalg.norm(dz_eff) == 0.0:
                if self._is_stationary(g_eff, z, fval):
                    self.sol = TerminatedPoint(
                        z=z.copy(),
                        x=x.copy(),
                        lam=lam.copy(),
                        nu=nu.copy(),
                        fval=fval
                    )
                    print("Stationary point reached!")
                    return StepStatus.TERMINATE
                t *= self.beta
                continue
            A_t = np.asarray(self.problem.eval_A(z_trial))
            b_t = np.asarray(self.problem.eval_b(z_trial)).flatten()
            P_t = np.asarray(self.problem.eval_P(z_trial))
            r_t = np.asarray(self.problem.eval_r(z_trial)).flatten()
            status_t, f_t, x_t, lam_t, nu_t = self.pdhg.solve(A_t, b_t, P_t, r_t, x0=self.x0)
            if status_t != PDHGStatus.OPTIMAL:
                print("Infeasible problem detected!")
                t *= self.beta
                continue
            self.x0 = x_t
            delta_m = -np.dot(g_eff, dz_eff)
            if abs(delta_m) >= self.tau:
                rho = (fval - f_t) / delta_m
                if rho >= self.eta:
                    accepted = True
                    break
            else:
                if f_t < fval - self.tau:
                    accepted = True
                    break
            print(f"Not accepted due to failure in rho or tau test.")
            t *= self.beta
        # step 6: post acceptance
        if not accepted:
            print("No acceptible step found. Requesting restart!")
            return StepStatus.RESTART
        # step 7: Trust Region update
        print(f"Step accepted with: {f_t}")
        grad_x_t = self.problem.grad_Lx(z_trial, x_t, lam_t, nu_t)
        active_t, _ = update_active_groups(
            grad_x_t, self.groups, self.pdhg.mu,
            active, self.eps_off
        )
        if abs(delta_m) >= self.tau:
            if rho > 1.0 - self.kappa and np.array_equal(active, active_t):
                self.delta *= self.amp
            else:
                self.delta *= self.damp
        else:
            self.delta *= self.damp
        # accept step
        self.z = z_trial
        self.active_prev = active_t.copy()
        return StepStatus.ACCEPTED
    
    def solve(self, z0):
        self.z = self._project_box(np.asarray(z0, dtype=float))
        self.delta = self.delta0
        self.active_prev = None
        for k in range(self.max_iter):
            print(f"\n--- ITER {k}, delta = {self.delta:.3e} ---")
            status = self.step()
            if status == StepStatus.ACCEPTED:
                continue
            if status == StepStatus.RESTART:
                print("Restart triggered.")
                self.restore_count += 1
                if (k >= self.switch_ratio * self.max_iter):
                    self.tol_mode = True
                z_rest, x, _, stat = self.restoration.solve(
                    y_k=self.z, z_init=self.z
                )
                if z_rest is None:
                    raise RuntimeError("Restoration failed.")
                self.z = self._project_box(z_rest)
                self.x0 = x
                self.delta = self.delta0
                self.active_prev = None
                continue
            if status == StepStatus.TERMINATE:
                break
        # print(self.min_g)
        return self.sol
