import numpy as np
from enum import Enum
from dataclasses import dataclass
from utils.group import InnerStatus
from utils.act import update_active_groups


class StepStatus(Enum):
    ACCEPTED = 0
    RESTART = 1
    TERMINATE = 2

class AcceptedType(Enum):
    SMOOTH = 0
    SWITCH = 1

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
            problem, inner_solver, retsoration_solver, groups,
            *, eta=0.1, beta=0.5, tau=1e-4, delta0=1.0,
            eps=1e-6, kappa=0.1, eps_off=1e-6,
            max_iter=1000, t_min=1e-8,
            damp=0.7, amp=1.5, switch_ratio=0.7,
            delta_warm=1e-2, warmup_feas_tol=1e-7,
            damp_warm=0.7, amp_warm=1.1, warmup_iters=30,
            gamma_f=1e-5, gamma_theta=1e-7, filter_max_size=50,
            memory_len=10, escape_radius=1e-3,
            switch_grad_ratio=0.9, verbose=False 
    ):
        """
        :param problem: BilevelProblem
        :param inner_solver: GroupPDHG/GroupSOCP
        :param retsoration_solver: RestorationNLP
        """
        self.problem = problem
        self.inner = inner_solver
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
        self.delta_warm = delta_warm
        self.warmup_feas_tol = warmup_feas_tol
        self.damp_warm = damp_warm
        self.amp_warm = amp_warm
        self.warmup_iters = warmup_iters
        self.gamma_f = gamma_f
        self.gamma_theta = gamma_theta
        self.filter_max_size = filter_max_size
        self.memory_len = memory_len
        self.escape_radius = escape_radius
        self.switch_grad_ratio = switch_grad_ratio
        self.verbose = verbose
        # internal global states
        self.restore_count = 0
        self.tol_mode = False
        self.min_g = np.inf
        self.active_prev = None
        self.sol = TerminatedPoint.empty()
        self.recent_z = []
        self.filter = []
        self.recent_switches = []
    
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

    def _constraint_violation(self, z, x):
        A = np.asarray(self.problem.eval_A(z))
        b = np.asarray(self.problem.eval_b(z)).flatten()
        P = np.asarray(self.problem.eval_P(z))
        r = np.asarray(self.problem.eval_r(z)).flatten()
        ineq = np.linalg.norm(np.maximum(A @ x - b, 0.0), np.inf)
        eq = np.linalg.norm(P @ x - r, np.inf)
        return ineq + eq
    
    def _rejects_filter(self, f_new, theta_new):
        for theta_old, f_old in self.filter:
            obj_improve = f_new >= f_old - self.gamma_f
            feas_improve = theta_new >= theta_old - self.gamma_theta
            rejected = obj_improve and feas_improve
            if rejected: return True
        return False
    
    def _update_filter(self, f_new, theta_new):
        new_filter = []
        for theta_old, f_old in self.filter:
            # remove entries dominated by new point
            dominated_by_new = (
                theta_new < theta_old - self.gamma_theta
                and
                f_new < f_old - self.gamma_f
            )
            if not dominated_by_new:
                new_filter.append((theta_old, f_old))
        new_filter.append((theta_new, f_new))
        if len(new_filter) > self.filter_max_size:
            new_filter = new_filter[-self.filter_max_size:]
        self.filter = new_filter
    
    def _recently_visited(self, z):
        for z_old in self.recent_z:
            if np.linalg.norm(z - z_old) < self.escape_radius:
                return True
        return False
    
    def _record_visit(self, z):
        self.recent_z.append(z.copy())
        if len(self.recent_z) > self.memory_len:
            self.recent_z.pop(0)

    def _record_switch(self, active_old, active_new):
        pair = (tuple(active_old.astype(int)), tuple(active_new.astype(int)))
        self.recent_switches.append(pair)
        if len(self.recent_switches) > self.memory_len:
            self.recent_switches.pop(0)
    
    def _recent_switch_repeated(self, active_old, active_new):
        pair = (tuple(active_old.astype(int)), tuple(active_new.astype(int)))
        return pair in self.recent_switches
    
    def _acceptance_test(self, active, active_t, same_active, delta_m, fval, f_t, theta, theta_t, z_trial, x_t, lam_t, nu_t, g_eff):
        accepted_type = None
        rho = 0.0
        accepted = False
        # same manifold
        if same_active:
            accepted = False
            if abs(delta_m) >= self.tau:
                rho = (fval - f_t) / delta_m
                if rho >= self.eta and not self._rejects_filter(
                    f_t, theta_t
                ):
                    accepted = True
            else:
                if not self._rejects_filter(
                    f_t, theta_t
                ):
                    accepted = True
            if accepted:
                accepted_type = AcceptedType.SMOOTH
        # different manifold
        else:
            g_trial = self.problem.grad_Lz(z_trial, x_t, lam_t, nu_t)
            g_trial_eff = self._effective_gradient(z_trial, g_trial)
            stationarity_improved = (
                np.linalg.norm(g_trial_eff)
                <
                self.switch_grad_ratio * np.linalg.norm(g_eff)
            )
            # obj_improved = (
            #     f_t < fval - self.switch_obj_tol
            # )
            filter_ok = not self._rejects_filter(f_t, theta_t)
            repeated = self._recent_switch_repeated(active, active_t)
            if filter_ok and stationarity_improved and not repeated:
                accepted = True
                accepted_type = AcceptedType.SWITCH
        return accepted, accepted_type, rho
                

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
        status, fval, x, lam, nu = self.inner.solve(A, b, P, r, x0=self.x0)
        if status != InnerStatus.OPTIMAL: raise RuntimeError(f"Inner solver failed with status {status}")
        self.last_x = x
        self.last_lam = lam
        self.last_nu = nu
        self.last_fval = fval
        ### constraint violation at starting
        theta = self._constraint_violation(z, x)
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
        ## for projected descent, use this
        # d = -g_eff / norm_g
        ## for trust region descent, use this
        d = np.clip(
            -self.delta * g_eff,
            self.problem.z_min - z,
            self.problem.z_max - z
        )
        # step 4: Group activation test
        grad_x = self.problem.grad_Lx(z, x, lam, nu)
        active, _ = update_active_groups(
            grad_x, self.groups, self.inner.mu, 
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
            # normal progression
            A_t = np.asarray(self.problem.eval_A(z_trial))
            b_t = np.asarray(self.problem.eval_b(z_trial)).flatten()
            P_t = np.asarray(self.problem.eval_P(z_trial))
            r_t = np.asarray(self.problem.eval_r(z_trial)).flatten()
            status_t, f_t, x_t, lam_t, nu_t = self.inner.solve(A_t, b_t, P_t, r_t, x0=self.x0)
            ## handle infeasibility
            if status_t != InnerStatus.OPTIMAL:
                print("Infeasible problem detected!")
                t *= self.beta
                continue
            ### constraint violation
            theta_t = self._constraint_violation(z_trial, x_t)
            delta_m = -np.dot(g_eff, dz_eff)
            grad_x_t = self.problem.grad_Lx(z_trial, x_t, lam_t, nu_t)
            ### active groups update
            active_t, _ = update_active_groups(
                grad_x_t, self.groups, self.inner.mu,
                active, self.eps_off
            )
            same_active = np.array_equal(active, active_t)
            accepted, accepted_type, rho = self._acceptance_test(active, active_t, same_active, delta_m, 
                                                            fval, f_t, theta, theta_t,
                                                            z_trial, x_t, lam_t, nu_t, g_eff)
            if accepted:
                self._update_filter(f_t, theta_t) 
                break
            print(f"Not accepted due to failure in rho or tau test.")
            t *= self.beta
        # step 6: post acceptance
        if not accepted:
            self._record_visit(z)
            print("No acceptible step found. Requesting restart!")
            return StepStatus.RESTART
        # step 7: Trust Region update
        print(f"Step accepted with: {f_t}, acceptance type: {accepted_type}")
        if accepted_type == AcceptedType.SWITCH:
            self._record_switch(active, active_t)
            self.delta *= self.damp
        else:
            if abs(delta_m) >= self.tau and rho > 1.0 - self.kappa:
                self.delta *= self.amp
            else:
                self.delta *= self.damp
        # accept step
        self.z = z_trial
        self.active_prev = active_t.copy()
        self.x0 = x_t
        self._record_visit(self.z)
        return StepStatus.ACCEPTED
    
    def _warm_start(self):
        prev = np.inf
        delta = self.delta_warm
        z0, x0 = self.z, self.last_x
        for wk in range(self.warmup_iters):
            z1, x1, _ = self.restoration.warm_start(z_ref=z0, x_ref=x0,
                                                        delta=delta)
            A_z = self.problem.A_sym(z1)
            b_z = self.problem.b_sym(z1)
            P_z = self.problem.P_sym(z1)
            r_z = self.problem.r_sym(z1)
            ineq_violation = np.linalg.norm(
                np.maximum(A_z @ x1 - b_z, 0.0), np.inf)
            eq_violation = np.linalg.norm(P_z @ x1 - r_z, np.inf)
            total_violation = ineq_violation + eq_violation
            print(
                f"    Warmup iter {wk}: ineq violation = {ineq_violation:.6f}, eq violation = {eq_violation:.6f}, delta = {delta:.3e}")
            if total_violation < self.warmup_feas_tol:
                print("Exiting warm up phase...")
                z0, x0 = z1, x1
                break
            if prev > total_violation:
                prev = total_violation
                delta *= self.damp_warm
                z0, x0 = z1, x1
            else:
                delta *= self.amp_warm
        return z0, x0
    
    def solve(self, z0, x0=None):
        if x0 is None: self.x0 = 0.5 * np.ones(self.problem.nx)
        else: self.x0 = x0
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
                z_rest, x = self._warm_start()
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
