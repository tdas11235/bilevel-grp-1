import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupPDHG


np.random.seed(5)
nx = 1000
nz = 100
groups = []
stride = 7
group_size = 40
for i in range(0, nx - group_size + 1, stride):
    groups.append(list(range(i, i + group_size)))
n_groups = len(groups)
c = np.random.uniform(-10, 10, nx)
mu = 5.0
lb_x = -100.0 * np.ones(nx)
ub_x = 100.0 * np.ones(nx)

# even larger problem : violates LICQ and solver fails to solve this
class Prob7(BilevelProblem):
    def __init__(self, nx=1000, nz=100, n_groups=150, mu=5.0):
        self.nx, self.nz = nx, nz
        self.m, self.l = 200, 100  # Increased constraints for N=1000
        self.mu = mu

        # 1. Adversarial Overlapping Groups
        self.groups = groups

        # 2. Ill-Conditioned Linear Operators
        # We use a decaying singular value structure for M1 and E1
        # to make the restoration and projection steps unstable.
        def ill_conditioned_matrix(rows, cols):
            mat = np.random.randn(rows, cols)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            s = np.logspace(0, -4, len(s))  # Condition number 1e4
            return (u * s) @ vh

        self.M1_np = ill_conditioned_matrix(self.m, self.nx)
        self.M2_np = np.random.randn(self.m, self.nz)

        self.E1_np = ill_conditioned_matrix(self.l, self.nx)
        # Make E1 have near-redundant rows to stress the Restoration Block
        for i in range(1, 10):
            self.E1_np[i, :] = self.E1_np[0, :] + \
                np.random.normal(0, 1e-4, self.nx)

        self.E2_np = np.random.randn(self.l, self.nz)

        # 3. Cost Vector c
        # Mixture of large negative and positive values to force x against bounds
        self.c_np = c

        # Convert to CasADi constants for the sym methods
        self.M1 = ca.DM(self.M1_np)
        self.M2 = ca.DM(self.M2_np)
        self.E1 = ca.DM(self.E1_np)
        self.E2 = ca.DM(self.E2_np)
        self.c = ca.DM(self.c_np)

        super().__init__(z_min=np.array([-10.0]*nz), z_max=np.array([10.0]*nz),
                         nx=nx, c=self.c_np, m=self.m, l=self.l)

    def A_sym(self, z):
        # Linear inequality coupling: M1*x <= M2*z - 5
        return self.M1

    def b_sym(self, z):
        # Dynamic RHS for inequalities
        return ca.mtimes(self.M2, z) - 5.0

    def P_sym(self, z):
        # Linear equality coupling: E1*x = E2*cos(z)
        return self.E1

    def r_sym(self, z):
        # Non-linear RHS: forces the upper-level to handle trig sensitivity
        return ca.mtimes(self.E2, ca.cos(z))


pdhg = GroupPDHG(
    c=c,
    groups=groups,
    mu=mu,
    lb_x=lb_x,
    ub_x=ub_x,
    tau=0.5,
    sigma=0.5,
    theta=0.5,
    max_iter=15000,
    tol=1e-3,
    tol_cons=1e-3,
    verbose=True
)

problem = Prob7()
restoration = RestorationNLP(problem=problem, rho=1.0, lb_x=lb_x, ub_x=ub_x)

solver = GAPTRSolver(
    problem=problem,
    pdhg_solver=pdhg,
    retsoration_solver=restoration,
    groups=groups,
    eta=0.1,
    beta=0.5,
    tau=1e-8,
    delta0=1.0,
    eps=5e-3,
    eps_off=1e-6,
    max_iter=100,
    amp=1.2,
    damp=0.9,
    verbose=False
)

z0 = 2.0 * np.ones(nz)
x0 = np.ones(nx) * 50
z0, x0 = restoration.warm_start(z0, x0, 10)
z0, x0, _, status = restoration.solve(y_k=z0, z_init=z0)
print(x0)
sol = solver.solve(z0, x0=x0)
print(sol)
