import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupPDHG


nx, nz = 4, 2
groups = [[0, 1], [2, 3]]
# mu = 0.01
mu = 1.0  # try this
# c = np.array([0.0, 0.0, 0.0, 0.0])
c = np.array([0.1, 0.1, 0.1, 0.1])    # try this

# switching constraints problem
class Prob3(BilevelProblem):
    def __init__(self, mu=1.0):
        nx, nz = 4, 2
        self.groups = [[0, 1], [2, 3]]
        self.c = c
        super().__init__(z_min=np.array([-5., -5.]), z_max=np.array([5., 5.]),
                         nx=nx, c=self.c, m=2, l=1)

    def A_sym(self, z):
        # Constraints form a V-shape in the feasible space depending on z
        row1 = ca.horzcat(1, z[0], 0, 0)
        row2 = ca.horzcat(1, -z[0], 0, 0)
        return ca.vertcat(row1, row2)

    def b_sym(self, z):
        return ca.vertcat(10 + z[1], 10 - z[1])

    def P_sym(self, z):
        return ca.horzcat(0, 0, 1, z[1]**2)

    def r_sym(self, z):
        return ca.vertcat(1.0)


pdhg = GroupPDHG(
    c=c,
    groups=groups,
    mu=mu,
    lb_x=-5.0 * np.ones(nx),
    ub_x=5.0 * np.ones(nx),
    tau=5e-2,
    sigma=5e-2,
    theta=0.5,
    max_iter=10000,
    tol=1e-6,
    tol_cons=1e-6,
    verbose=False
)

problem = Prob3()
restoration = RestorationNLP(problem=problem, rho=0.001)

solver = GAPTRSolver(
    problem=problem,
    pdhg_solver=pdhg,
    retsoration_solver=restoration,
    groups=groups,
    eta=0.1,
    beta=0.5,
    tau=1e-8,
    delta0=1.0,
    eps=5e-6,
    eps_off=1e-6,
    max_iter=200,
    amp=1.2,
    damp=0.9,
    verbose=False
)

z0 = np.array([-1.0, 1.0])

sol = solver.solve(z0)
print(sol)
