import numpy as np
import casadi as ca


class BilevelProblem:
    def __init__(self, z_min, z_max, nx, m):
        """
        q_min, q_max : box constraints on q
        nx           : dimension of x
        m            : number of constraints (rows of A)
        """
        self.z_min = np.array(z_min)
        self.z_max = np.array(z_max)
        self.nz = len(z_min)
        self.nx = nx
        self.m = m
        # build symbolic structure
        self._build_symbolic()
        self._build_grad_L()
        self._build_dphi_fun()

    def A_sym(self, z):
        """
        Symbolic A(z) of dims m-by-nx
        Must be overridden by user
        """
        raise NotImplementedError

    def b_sym(self, z):
        """
        Symbolic b(z) of dim m
        Must be overridden by user
        """
        raise NotImplementedError
    
    def P_sym(self, z):
        """
        Symbolic P(z) of dim m
        Must be overridden by user
        """
        raise NotImplementedError
    
    def r_sym(self, z):
        """
        Symbolic r(z) of dim m
        Must be overridden by user
        """
        raise NotImplementedError

    def _build_symbolic(self):
        z = ca.MX.sym("z", self.nz)
        A = self.A_sym(z)
        b = self.b_sym(z)
        P = self.P_sym(z)
        r = self.r_sym(z)
        self._A_fun = ca.Function("A_fun", [z], [A])
        self._b_fun = ca.Function("b_fun", [z], [b])
        self._P_fun = ca.Function("P_fun", [z], [P])
        self._r_fun = ca.Function("r_fun", [z], [r])

    def eval_A(self, z):
        """
        Return A(z)
        """
        return self._A_fun(z)

    def eval_b(self, z):
        """
        Return b(z)
        """
        return self._b_fun(z)
    
    def eval_P(self, z):
        """
        Return P(z)
        """
        return self._P_fun(z)

    def eval_r(self, z):
        """
        Return r(z)
        """
        return self._r_fun(z)

    def _build_grad_L(self):
        """
        Build casadi function for gradient of lagrangian
        """
        q = ca.MX.sym('q', self.nq)
        x = ca.MX.sym('x', self.nx)
        dual = ca.MX.sym('dual', self.m)
        A = self.A_sym(q)
        b = self.b_sym(q)
        expr = ca.dot(dual, ca.mtimes(A, x) - b)
        grad = ca.gradient(expr, q)
        self._grad_L_fun = ca.Function("grad_L",
                                       [q, x, dual], [grad])

    def grad_L(self, q, x, dual):
        """
        Return gradient wrt q:
        dual.T @ (A(q) @ x - b(q))
        """
        g = self._grad_L_fun(q, x, dual)
        return np.asarray(g).flatten()

    def _build_dphi_fun(self):
        """
        Build casadi function for directional derivative of phi(q)
        """
        q = ca.MX.sym("q", self.nq)
        x = ca.MX.sym("x", self.nx)
        d = ca.MX.sym("d", self.nq)
        A = self.A_sym(q)
        b = self.b_sym(q)
        phi = ca.mtimes(A, x) - b
        dphi = ca.jacobian(phi, q) @ d
        self._dphi_fun = ca.Function("dphi", [q, x, d], [dphi])
