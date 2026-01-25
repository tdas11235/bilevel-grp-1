import numpy as np
import casadi as ca


class BilevelProblem:
    def __init__(self, z_min, z_max, nx, c, m, l):
        """
        z_min, z_max : box constraints on z
        nx           : dimension of x
        m            : number of inequality constraints (rows of A)
        l            : number of equality constraints (rows of P)
        """
        self.z_min = np.array(z_min)
        self.z_max = np.array(z_max)
        self.nz = len(z_min)
        self.nx = nx
        self.c = c
        self.m = m
        self.l = l
        # build symbolic structure
        self._build_symbolic()
        self._build_grad_Lz()
        self._build_grad_Lx()

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
        Symbolic P(z) of dim l
        Must be overridden by user
        """
        raise NotImplementedError
    
    def r_sym(self, z):
        """
        Symbolic r(z) of dim l
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

    def _build_grad_Lz(self):
        """
        Build casadi function for gradient of lagrangian
        wrt z
        """
        z = ca.MX.sym('z', self.nz)
        x = ca.MX.sym('x', self.nx)
        lam = ca.MX.sym('lam', self.m)
        nu = ca.MX.sym('nu', self.l)
        A = self.A_sym(z)
        b = self.b_sym(z)
        P = self.P_sym(z)
        r = self.r_sym(z)
        expr1 = ca.dot(lam, ca.mtimes(A, x) - b)
        expr2 = ca.dot(nu, ca.mtimes(P, x) - r)
        grad = ca.gradient(expr1+expr2, z)
        self._grad_Lz_fun = ca.Function("grad_Lz",
                                       [z, x, lam, nu], [grad])

    def grad_Lz(self, z, x, lam, nu):
        """
        Return gradient wrt z:
        lam.T @ (A(z) @ x - b(z)) + nu.T @ (P(z) @ x - r(z))
        """
        g = self._grad_Lz_fun(z, x, lam, nu)
        return np.asarray(g).flatten()

    def _build_grad_Lx(self):
        """
        Build casadi function for gradient of smooth Lagrangian
        wrt x
        """
        z = ca.MX.sym('z', self.nz)
        x = ca.MX.sym('x', self.nx)
        lam = ca.MX.sym('lam', self.m)
        nu = ca.MX.sym('nu', self.l)
        A = self.A_sym(z)
        P = self.P_sym(z)
        grad_x = (
            ca.MX(self.c)
            + ca.mtimes(A.T, lam)
            + ca.mtimes(P.T, nu)
        )
        self._grad_Lx_fun = ca.Function(
            "grad_Lx",
            [z, x, lam, nu],
            [grad_x]
        )
    
    def grad_Lx(self, z, x, lam, nu):
        """
        Return gradient wrt x of smooth Lagrangian
        """
        g = self._grad_Lx_fun(z, x, lam, nu)
        return np.asarray(g).flatten()
