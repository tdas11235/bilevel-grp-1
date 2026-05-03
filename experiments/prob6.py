import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupSOCP, GroupPDHG
import time

# mu=1.0, tol, cons_tol=1e-3 for pdhg
mu = 10.0
np.random.seed(0)
nx = 500
nz = 50
# mu = 1.0
n_groups = 50
groups = []
step = 10
c = np.random.uniform(-1.0, 0.5, nx)
groups = [list(range(i*10, min(i*10 + 10, nx))) for i in range(n_groups)]
lb_x = -5.0 * np.ones(nx)
ub_x = 5.0 * np.ones(nx)

print(c)

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


# pdhg = GroupSOCP(
#     c=c,
#     groups=groups,
#     mu=mu,
#     lb_x=lb_x,
#     ub_x=ub_x,
#     max_iters=10000,
#     verbose=False,
    
# )


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
    inner_solver=pdhg,
    restoration_solver=restoration,
    groups=groups,
    eta=0.1,
    beta=0.5,
    tau=1e-8,
    delta0=1.0,
    eps=5e-3,
    eps_off=1e-6,
    max_iter=200,
    amp=1.2,
    damp=0.9,
    escape_radius=4e-2,
    verbose=False
)

z0 = -0.5 * np.ones(nz)
x0 = np.ones(nx)
delta = 1e1
prev = np.inf
for k in range(200):
    z1, x1, _ = restoration.warm_start(z_ref=z0, x_ref=x0,
                                            delta=delta)
    A_z = problem.A_sym(z1)
    b_z = problem.b_sym(z1)
    P_z = problem.P_sym(z1)
    r_z = problem.r_sym(z1)
    ineq_violation = np.linalg.norm(
        np.maximum(A_z @ x1 - b_z, 0.0), np.inf)
    eq_violation = np.linalg.norm(P_z @ x1 - r_z, np.inf)
    total_violation = ineq_violation + eq_violation
    print(
        f"    Warmup iter {k}: ineq violation = {ineq_violation:.6f}, eq violation = {eq_violation:.6f}, delta = {delta:.3e}")
    if total_violation < 1e-7:
        print("Exiting warm up phase...")
        z0, x0 = z1, x1
        break
    if prev > total_violation:
        prev = total_violation
        delta *= 0.5
        z0, x0 = z1, x1
    else:
        delta *= 1.2

start = time.time()
sol = solver.solve(z0, x0=x0)
print(sol)
print(f"Time taken: {time.time() - start}")
