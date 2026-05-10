import casadi as ca
import numpy as np
from highspy import Highs
import highspy as hp
import scipy.sparse as sp


class RestorationNLP:
    def __init__(
            self, problem, rho,
            lb_x=None, ub_x=None,
            verbose=True
        ):
        """
        problem:    BilevelProblem instance
        rho:        Regularization parameter
        """
        self.problem = problem
        self.rho = rho
        self.verbose = verbose
        nx = self.problem.nx
        if lb_x is None: self.lb_x = np.zeros(nx)
        else: self.lb_x = lb_x = np.asarray(lb_x).reshape(-1)
        if ub_x is None: self.ub_x = np.inf * np.ones(nx)
        else: self.ub_x = ub_x = np.asarray(ub_x).reshape(-1)
        self._build_solver()

    def _build_solver(self):
        nz = self.problem.nz
        nx = self.problem.nx
        m = self.problem.m
        l = self.problem.l
        # decision vars
        z = ca.MX.sym("z", nz)
        x = ca.MX.sym("x", nx)
        e = ca.MX.sym("e")
        # parameter (stability center)
        y = ca.MX.sym("y", nz)
        # expressions
        A_z = self.problem.A_sym(z)
        b_z = self.problem.b_sym(z)
        P_z = self.problem.P_sym(z)
        r_z = self.problem.r_sym(z)
        # residuals
        v = ca.mtimes(A_z, x) - b_z
        u = ca.mtimes(P_z, x) - r_z
        # jacobians
        Jv = ca.jacobian(v, z)   # shape (m, nz)
        Ju = ca.jacobian(u, z)   # shape (l, nz)
        # compile functions
        self.v_fun = ca.Function("v_fun", [z, x], [v])
        self.u_fun = ca.Function("u_fun", [z, x], [u])
        self.Jv_fun = ca.Function("Jv_fun", [z, x], [Jv])
        self.Ju_fun = ca.Function("Ju_fun", [z, x], [Ju])
        self.A_fun = ca.Function("A_fun", [z], [A_z])
        self.P_fun = ca.Function("P_fun", [z], [P_z])
        # constraints
        cons = []
        cons += [ca.mtimes(A_z, x) - b_z - e * ca.DM.ones(m)]
        cons += [ca.mtimes(P_z, x) - r_z]
        # objective
        obj = e + (self.rho / 2.0) * ca.sumsqr(z - y)
        # bounds
        self.lbg = np.concatenate([
            -np.inf * np.ones(m),
            np.zeros(l)
        ])
        self.ubg = np.concatenate([
            np.zeros(m),
            np.zeros(l)
        ])
        self.lbx = np.concatenate([
            self.problem.z_min,
            self.lb_x,
            [-np.inf]
        ])
        self.ubx = np.concatenate([
            self.problem.z_max,
            self.ub_x,
            [0.0]
        ])
        self.nx = nx
        self.nz = nz

    def warm_start(self, z_ref, x_ref, delta=1e-2):
        if delta is None: delta = 1e-2
        nz, nx = self.nz, self.nx
        m, l = self.problem.m, self.problem.l
        nvar = nz + nx + 1  # [dz, dx, t]
        # --- evaluate ---
        v = self.v_fun(z_ref, x_ref).full().squeeze()
        u = self.u_fun(z_ref, x_ref).full().squeeze()
        Jv = self.Jv_fun(z_ref, x_ref).full()
        Ju = self.Ju_fun(z_ref, x_ref).full()
        A = self.A_fun(z_ref).full()
        P = self.P_fun(z_ref).full()

        highs = hp.Highs()
        # highs.silent()
        # --- bounds (trust region and feasibility) ---
        dz_lb = np.maximum(self.problem.z_min - z_ref, -delta)
        dz_ub = np.minimum(self.problem.z_max - z_ref,  delta)
        dx_lb = np.maximum(self.lb_x - x_ref, -delta)
        dx_ub = np.minimum(self.ub_x - x_ref,  delta)
        lb = np.concatenate([dz_lb, dx_lb, [0.0]])
        ub = np.concatenate([dz_ub, dx_ub, [np.inf]])
        highs.addVars(nvar, lb.astype(np.float64), ub.astype(np.float64))

        # --- objective: min t ---
        idx = np.arange(nvar, dtype=np.int32)
        cost = np.zeros(nvar, dtype=np.float64)
        cost[-1] = 1.0
        highs.changeColsCost(nvar, idx, cost)
        # --- constraints ---
        # v <= t
        for i in range(m):
            row_idx = []
            row_val = []
            # dz part
            for j in range(nz):
                val = Jv[i, j]
                if val != 0.0:
                    row_idx.append(j)
                    row_val.append(val)
            # dx part
            for j in range(nx):
                val = A[i, j]
                if val != 0.0:
                    row_idx.append(nz + j)
                    row_val.append(val)
            # t
            row_idx.append(nz + nx)
            row_val.append(-1.0)
            highs.addRow(
                -np.inf,
                -v[i],
                len(row_idx),
                np.array(row_idx, dtype=np.int32),
                np.array(row_val, dtype=np.float64)
            )

        # u <= t
        for i in range(l):
            row_idx = []
            row_val = []
            for j in range(nz):
                val = Ju[i, j]
                if val != 0.0:
                    row_idx.append(j)
                    row_val.append(val)
            for j in range(nx):
                val = P[i, j]
                if val != 0.0:
                    row_idx.append(nz + j)
                    row_val.append(val)
            row_idx.append(nz + nx)
            row_val.append(-1.0)
            highs.addRow(
                -np.inf,
                -u[i],
                len(row_idx),
                np.array(row_idx, dtype=np.int32),
                np.array(row_val, dtype=np.float64)
            )
        # -u <= t
        for i in range(l):
            row_idx = []
            row_val = []
            for j in range(nz):
                val = -Ju[i, j]
                if val != 0.0:
                    row_idx.append(j)
                    row_val.append(val)
            for j in range(nx):
                val = -P[i, j]
                if val != 0.0:
                    row_idx.append(nz + j)
                    row_val.append(val)
            row_idx.append(nz + nx)
            row_val.append(-1.0)
            highs.addRow(
                -np.inf,
                u[i],
                len(row_idx),
                np.array(row_idx, dtype=np.int32),
                np.array(row_val, dtype=np.float64)
            )

        # --- solve ---
        highs.silent()
        highs.run()
        status = highs.getModelStatus()
        if status != hp.HighsModelStatus.kOptimal:
            return z_ref, x_ref, 0
        sol = np.array(highs.getSolution().col_value)
        dz = sol[:nz]
        dx = sol[nz:nz+nx]
        t_pred = sol[-1]
        return z_ref + dz, x_ref + dx, t_pred
    
    def _soc_rhs(self, z0, x0, z1, x1, delta):
        nx = self.nx
        m, l = self.problem.m, self.problem.l
        # Jacobians at ORIGINAL point
        A0 = self.A_fun(z0).full()
        P0 = self.P_fun(z0).full()
        # TRUE residuals at trial point
        v_true = self.v_fun(z1, x1).full().squeeze()
        u_true = self.u_fun(z1, x1).full().squeeze()
        highs = hp.Highs()
        highs.silent()
        nvar = nx + 1  # [dx_soc, t]
        # bounds
        dx_lb = -delta * np.ones(nx)
        dx_ub = delta * np.ones(nx)
        lb = np.concatenate([dx_lb, [0.0]])
        ub = np.concatenate([dx_ub, [np.inf]])
        highs.addVars(nvar, lb.astype(np.float64), ub.astype(np.float64))
        # objective: min t
        cost = np.zeros(nvar)
        cost[-1] = 1.0
        highs.changeColsCost(nvar, np.arange(nvar, dtype=np.int32), cost)
        # A0 dx <= -v_true + t
        for i in range(m):
            row_idx = []
            row_val = []
            for j in range(nx):
                if A0[i, j] != 0.0:
                    row_idx.append(j)
                    row_val.append(A0[i, j])
            row_idx.append(nx)
            row_val.append(-1.0)
            highs.addRow(
                -np.inf,
                -v_true[i],
                len(row_idx),
                np.array(row_idx, dtype=np.int32),
                np.array(row_val, dtype=np.float64)
            )
        # P0 dx = -u_true
        for i in range(l):
            row_idx = []
            row_val = []
            for j in range(nx):
                if P0[i, j] != 0.0:
                    row_idx.append(j)
                    row_val.append(P0[i, j])
            highs.addRow(
                -u_true[i],
                -u_true[i],
                len(row_idx),
                np.array(row_idx, dtype=np.int32),
                np.array(row_val, dtype=np.float64)
            )
        highs.run()
        if highs.getModelStatus() != hp.HighsModelStatus.kOptimal:
            return x1, np.inf
        sol = np.array(highs.getSolution().col_value)
        dx_soc = sol[:nx]
        return x1 + dx_soc, sol[-1]

    def solve(self, y_k, z_init=None, x_init=None, e_init=0.0):
        """
        Solve the restoration phase centered at y_k
        """
        if z_init is None:
            z_init = y_k
        if x_init is None:
            x_init = np.ones(self.nx)
        x0 = np.concatenate([z_init, x_init, [e_init]])
        # solve
        sol = self.solver(
            x0=x0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=y_k
        )
        stats = self.solver.stats()
        status = stats.get("return_status", "Unknown")
        print(status)
        if status not in ("Solve_Succeeded", "Converged_To_Acceptable_Point"):
            return None, None, None, status
        w = sol["x"].full().squeeze()
        z = w[:self.nz]
        x = w[self.nz: self.nz + self.nx]
        e = w[-1]
        return z, x, e, status
