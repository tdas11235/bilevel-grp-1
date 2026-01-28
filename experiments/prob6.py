import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupPDHG

# mu=1.0, tol, cons_tol=1e-3 for pdhg
mu = 5.0
np.random.seed(0)
nx = 500
nz = 50
# mu = 1.0
n_groups = 50
groups = []
step = 10
c = np.random.uniform(-1.0, 0.5, nx)
groups = [list(range(i*10, min(i*10 + 15, nx))) for i in range(n_groups)]
lb_x = -5.0 * np.ones(nx)
ub_x = 5.0 * np.ones(nx)

# large scale problem
class Prob6(BilevelProblem):
    def __init__(self, nx=500, nz=50, n_groups=50, mu=0.1):
        self.nx, self.nz = nx, nz
        self.m, self.l = 100, 50 # 100 inequalities, 50 equalities
        
        # Overlapping groups logic (same as before)
        self.groups = groups
        
        # Fixed random matrices to create dense coupling
        self.M1 = np.random.randn(self.m, nx)
        self.M2 = np.random.randn(self.m, nz)
        self.E1 = np.random.randn(self.l, nx)
        self.E2 = np.random.randn(self.l, nz)
        
        # c has some negative values so x wants to grow large
        self.c = c
        
        super().__init__(z_min=np.array([-1.0]*nz), z_max=np.array([1.0]*nz),
                         nx=nx, c=self.c, m=self.m, l=self.l)

    def A_sym(self, z):
        return self.M1 

    def b_sym(self, z):
        return ca.mtimes(self.M2, z) - 5.0 

    def P_sym(self, z):
        return self.E1

    def r_sym(self, z):
        return ca.mtimes(self.E2, ca.cos(z))


pdhg = GroupPDHG(
    c=c,
    groups=groups,
    mu=mu,
    lb_x=lb_x,
    ub_x=ub_x,
    tau=5e-3,
    sigma=5e-3,
    theta=0.5,
    max_iter=10000,
    tol=1e-5,
    tol_cons=1e-5,
    verbose=True
)

problem = Prob6()
restoration = RestorationNLP(problem=problem, rho=10.0, lb_x=lb_x, ub_x=ub_x)

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

z0 = -0.5 * np.ones(nz)
z0, x0, _, status = restoration.solve(y_k=z0, z_init=z0)
sol = solver.solve(z0, x0=x0)
print(sol)
