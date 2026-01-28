import casadi as ca
import numpy as np


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
        # solver objects
        nlp = {
            "x": ca.vertcat(z, x, e),
            "f": obj,
            "g": ca.vertcat(*cons),
            "p": y
        }
        if self.verbose:
            opts = {
                "ipopt.print_level": 5,
                "print_time": True
            }
        else:
            opts = {
                "ipopt.print_level": 0,
                "print_time": False
            }
        self.solver = ca.nlpsol("restoration", "ipopt", nlp, opts)
        self.nx = nx
        self.nz = nz

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
