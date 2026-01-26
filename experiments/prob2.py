import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupPDHG


nx = 5
nz = 2


class Prob2(BilevelProblem):
    def __init__(self, mu=1.0):
        nx = 5
        nz = 2
        self.groups = [[0, 1, 2], [2, 3, 4]]
        self.c = np.array([1.0, 1.1, 0.5, 1.2, 1.0])
        super().__init__(z_min=np.array([-2.0, -2.0]), z_max=np.array([2.0, 2.0]),
                         nx=nx, c=self.c, m=2, l=1)

    def A_sym(self, z):
        row1 = ca.horzcat(z[0], 1, 0, 0, 0)
        row2 = ca.horzcat(0, 0, 0, 1, z[0])
        return ca.vertcat(row1, row2)

    def b_sym(self, z):
        return ca.vertcat(5.0, 5.0)

    def P_sym(self, z):
        return ca.horzcat(0, (z[0]**2 + z[1]), 0, -1.0, 0)
        # return ca.horzcat(0, (1.0 + z[1]), 0, -1.0, 0)

    def r_sym(self, z):
        return ca.vertcat(1.0 - 0.5 * z[0])


groups = [[0, 1, 2], [2, 3, 4]]

mu = 10

pdhg = GroupPDHG(
    c=np.ones(nx),
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

problem = Prob2()
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