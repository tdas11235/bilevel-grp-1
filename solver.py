import cvxpy as cp
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


@dataclass
class AcceptablePoint:
    z: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    nu: np.ndarray
    fval: float
    theta: float
    grad_norm: float
    iter_idx: int

    @classmethod
    def empty(cls):
        return cls(
            z=np.array([np.nan]),
            x=np.array([np.nan]),
            lam=np.array([np.nan]),
            nu=np.array([np.nan]),
            fval=np.inf,
            theta=np.inf,
            grad_norm=np.inf,
            iter_idx=-1
        )

class GAPTRSolver:
    def __init__(
            self,
            problem, inner_solver, restoration_solver, groups,
            *, eta=0.1, beta=0.5, tau=1e-4, delta0=1.0,
            eps=1e-6, kappa=0.1, eps_off=1e-6,
            max_iter=1000, t_min=1e-8,
            damp=0.7, amp=1.5, switch_ratio=0.7,
            delta_warm=1e-2, warmup_feas_tol=1e-7,
            damp_warm=0.7, amp_warm=1.1, warmup_iters=100,
            gamma_f=1e-5, gamma_theta=1e-7, filter_max_size=50,
            memory_len=10, escape_radius=1e-3,
            switch_grad_ratio=0.9, 
            boundary_tol=1e-3, grad_memory_len=20, persist_threshold=4,
            grad_boundary_tol=1e-4, perturb_chances=3, verbose=False 
    ):
        """
        :param problem: BilevelProblem
        :param inner_solver: GroupPDHG/GroupSOCP
        :param retsoration_solver: RestorationNLP
        """
        self.problem = problem
        self.inner = inner_solver
        self.restoration = restoration_solver
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
        self.boundary_tol = boundary_tol
        self.grad_memory_len = grad_memory_len
        self.persist_threshold = persist_threshold
        self.grad_boundary_tol = grad_boundary_tol
        self.perturb_chances = perturb_chances
        self.verbose = verbose
        # internal global states
        self.restore_count = 0
        self.stuck_count = 0
        self.tol_mode = False
        self.min_g = np.inf
        self.active_prev = None
        self.sol = TerminatedPoint.empty()
        self.recent_z = []
        self.recent_restore = []
        self.filter = []
        self.recent_switches = []
        self.recent_gradients = []
        self.manifold_persist_count = 0
        self.acceptable_points = []
        self.last_restart_basin = None
    
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
    
    def _record_acceptable_point(
        self, z, x, lam, nu, fval, theta, g_eff, k
    ):
        grad_norm = np.linalg.norm(g_eff)
        check = grad_norm / max(1.0, np.abs(fval), np.linalg.norm(z)) < self.eps
        if check:
            pt = AcceptablePoint(
                z=z.copy(),
                x=x.copy(),
                lam=lam.copy(),
                nu=nu.copy(),
                fval=fval,
                theta=theta,
                grad_norm=grad_norm,
                iter_idx=k
            )
            self.acceptable_points.append(pt)
    
    def _near_switching_boundary(self, active, gnorms):
        """
        Returns True if at least one group is switches
        Based on one-sided hysteresis of (mu - eps_off) deactivation
        """
        for i, gn in enumerate(gnorms):
            if active[i]:
                # close to deactivation boundary
                if abs(gn - (self.inner.mu - self.eps_off)) <= self.boundary_tol:
                    return True
            else:
                # close to activation boundary
                if abs(gn - self.inner.mu) <= self.boundary_tol:
                    return True
        return False
    
    def _record_gradient(self, g_eff, same_active):
        """
        Store necessary manifold gradients only
        """
        if same_active:
            self.manifold_persist_count += 1
        else:
            self.manifold_persist_count = 0
        small_grad = (
            np.linalg.norm(g_eff) <= self.grad_boundary_tol
        )
        if (
            self.manifold_persist_count >= self.persist_threshold
            or small_grad
        ):
            self.recent_gradients.append(g_eff.copy())
            if len(self.recent_gradients) > self.grad_memory_len:
                self.recent_gradients.pop(0)
        
    def _minimum_norm_subgradient(self, g_current):
        """
        Minimum gradient from history based convex hull
        Solve:

        min || sum_i alpha_i g_i ||^2
        s.t.
            alpha_i >= 0
            sum alpha_i = 1

        Small dense QP in gradient space.
        """
        grads = [g_current.copy()]
        for g in self.recent_gradients:
            grads.append(g.copy())
        G = np.column_stack(grads)   # shape: (nz, m)
        m = G.shape[1]
        # QP:
        # min 0.5 * alpha^T H alpha
        # where H = G^T G
        H = G.T @ G + 1e-10 * np.eye(m)
        try:
            alpha = cp.Variable(m)
            objective = cp.Minimize(
                0.5 * cp.quad_form(alpha, H)
            )
            constraints = [
                alpha >= 0,
                cp.sum(alpha) == 1
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(warm_start=True, verbose=False)
            if alpha.value is None:
                return g_current.copy()
            alpha_val = np.asarray(alpha.value).flatten()
            g_min = G @ alpha_val
            return g_min
        except Exception:
            # safe fallback
            return g_current.copy()

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
        m = np.inf
        for z_old in self.recent_z:
            m = min(m, np.linalg.norm(z - z_old) /
                    max(1.0, np.linalg.norm(z)))
            if np.linalg.norm(z - z_old) / max(1.0, np.linalg.norm(z)) < self.escape_radius:
                return True
        print(m)
        return False
    
    def _recently_restored(self, z):
        m = np.inf
        for z_old in self.recent_restore:
            m = min(m, np.linalg.norm(z - z_old) /
                    max(1.0, np.linalg.norm(z)))
            if np.linalg.norm(z - z_old) / max(1.0, np.linalg.norm(z)) < self.escape_radius:
                return True
        print(m)
        return False
    
    def _record_visit(self, z):
        self.recent_z.append(z.copy())
        if len(self.recent_z) > self.memory_len:
            self.recent_z.pop(0)

    def _record_restore(self, z):
        self.recent_restore.append(z.copy())
        if len(self.recent_restore) > self.memory_len:
            self.recent_restore.pop(0)

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
        # step 3: Group activation test
        grad_x = self.problem.grad_Lx(z, x, lam, nu)
        active, gnorms = update_active_groups(
            grad_x, self.groups, self.inner.mu,
            self.active_prev, self.eps_off
        )
        same_active_now = (
            self.active_prev is not None
            and np.array_equal(active, self.active_prev)
        )
        self._record_gradient(
            g_eff,
            same_active_now
        )
        ## minimum norm subgrad for manifold boundaries
        if self._near_switching_boundary(active, gnorms):
            g_eff = self._minimum_norm_subgradient(g_eff)
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
        # step 4: descent direction
        norm_g = np.linalg.norm(g_eff)
        self.min_g = min(self.min_g, norm_g)
        ## for projected descent, use this
        d = -g_eff / norm_g
        ## for trust region descent, use this
        # d = np.clip(
        #     -self.delta * g_eff,
        #     self.problem.z_min - z,
        #     self.problem.z_max - z
        # )
        # step 5: Acceptance loop
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
        self._record_acceptable_point(
            z, x, lam, nu, fval, theta, g_eff, self.current_iter
        )
        self.stuck_count = 0
        return StepStatus.ACCEPTED
    
    def _warm_start(self):
        prev = np.inf
        delta = self.delta_warm
        z0, x0 = self.z, self.last_x
        delta_min = getattr(self, "delta_min", 1e-8)
        delta_max = getattr(self, "delta_max", 1e2)
        stall_count = 0
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
            improvement = prev - total_violation
            improved = total_violation < prev
            stalled = improvement < 1e-5
            print(
                f"    Warmup iter {wk}: "
                f"ineq = {ineq_violation:.6f}, "
                f"eq = {eq_violation:.6f}, "
                f"total = {total_violation:.6f}, "
                f"delta = {delta:.3e}"
            )
            # convergence
            # ---- accept iterate if better ----
            if improved:
                z0, x0 = z1, x1
                prev = total_violation
            # ---- update stall counter ----
            if stalled:
                stall_count += 1
            else:
                stall_count = 0
            # ---- delta update logic ----
            if improved and not stalled:
                # good progress -> shrink
                delta *= self.damp_warm
            else:
                # either no improvement OR weak improvement -> expand
                delta *= self.amp_warm
            delta = np.clip(delta, delta_min, delta_max)
        return z0, x0
    
    def _best_acceptable_point(self):
        if not self.acceptable_points:
            return None
        # first minimum fval, then minimum theta
        best = min(
            self.acceptable_points,
            key=lambda p: (p.grad_norm, p.fval)
        )
        return best
    
    def _escape_perturb(self, z, attempt):
        noise = np.random.randn(*z.shape)
        noise_norm = np.linalg.norm(noise)
        if noise_norm == 0:
            return z
        noise /= noise_norm
        if self.recent_gradients:
            g = self.recent_gradients[-1]
            g = g / (np.linalg.norm(g) + 1e-12)
            noise = noise - 0.7 * g
            noise /= np.linalg.norm(noise)
        alpha = self.delta0 * 2 ** (attempt + 1)
        z_new = z + alpha * noise
        return self._project_box(z_new)
    
    def solve(self, z0, x0=None):
        if x0 is None: self.x0 = 0.5 * np.ones(self.problem.nx)
        else: self.x0 = x0
        self.z = self._project_box(np.asarray(z0, dtype=float))
        self.delta = self.delta0
        self.active_prev = None
        for k in range(self.max_iter):
            print(f"\n--- ITER {k}, delta = {self.delta:.3e} ---")
            self.current_iter = k
            status = self.step()
            if status == StepStatus.ACCEPTED:
                continue
            if status == StepStatus.RESTART:
                self.stuck_count += 1
                print("Restart triggered.")
                if (k >= self.switch_ratio * self.max_iter):
                    self.tol_mode = True
                self.restore_count += 1
                z_rest, x = self._warm_start()
                if z_rest is None:
                    raise RuntimeError("Restoration failed.")
                # try noise perturbation to escape the basin
                stuck = (
                    self._recently_visited(z_rest) or
                    self._recently_restored(z_rest)
                ) and self.stuck_count >= 2
                if stuck:
                    print("Failed to find new point. Starting noise perturbation ...")
                    A = np.asarray(self.problem.eval_A(z_rest))
                    b = np.asarray(self.problem.eval_b(z_rest)).flatten()
                    P = np.asarray(self.problem.eval_P(z_rest))
                    r = np.asarray(self.problem.eval_r(z_rest)).flatten()
                    status, _, x_rest, lam_rest, nu_rest = self.inner.solve(
                        A, b, P, r, x0=x
                    )
                    grad_x_rest = self.problem.grad_Lx(z_rest, x_rest, lam_rest, nu_rest)
                    active_rest, _ = update_active_groups(
                        grad_x_rest, self.groups, self.inner.mu,
                        None, self.eps_off
                    )
                    self.last_restart_basin = active_rest.copy()
                    for attempt in range(self.perturb_chances):
                        self.z = self._escape_perturb(z_rest, attempt)
                        self.x0 = 0.5 * np.ones(self.problem.nx)
                        z_rest, x = self._warm_start()
                        if z_rest is None:
                            raise RuntimeError("Restoration failed.")
                        A = np.asarray(self.problem.eval_A(z_rest))
                        b = np.asarray(self.problem.eval_b(z_rest)).flatten()
                        P = np.asarray(self.problem.eval_P(z_rest))
                        r = np.asarray(self.problem.eval_r(z_rest)).flatten()
                        status, _, x_rest, lam_rest, nu_rest = self.inner.solve(
                            A, b, P, r, x0=x
                        )
                        grad_x_rest = self.problem.grad_Lx(
                            z_rest, x_rest, lam_rest, nu_rest)
                        active_rest, _ = update_active_groups(
                            grad_x_rest, self.groups, self.inner.mu,
                            None, self.eps_off
                        )
                        same_basin = (
                            self.last_restart_basin is not None
                            and np.array_equal(active_rest, self.last_restart_basin)
                        )
                        if not (self._recently_visited(z_rest) or self._recently_restored(z_rest)) and not same_basin:
                            break
                    else:
                        print("Failed to escape basin ...")
                        break
                self.z = self._project_box(z_rest)
                self._record_restore(self.z)
                self.x0 = x
                self.delta = self.delta0
                self.active_prev = None
                continue
            if status == StepStatus.TERMINATE:
                break
        # print(self.min_g)
        # return self.sol
        if not np.isnan(self.sol.fval):
            return self.sol
        # fallback to best acceptable point
        best = self._best_acceptable_point()
        if best is not None:
            print(
                f"Returning best acceptable point "
                f"(iter={best.iter_idx}, "
                f"theta={best.theta:.3e}, "
                f"grad_norm={best.grad_norm:.3e}"
                f"f={best.fval:.6f})"
            )
            return TerminatedPoint(
                z=best.z,
                x=best.x,
                lam=best.lam,
                nu=best.nu,
                fval=best.fval
            )
        raise RuntimeError(
            "Solver terminated without stationary or acceptable point."
        )
