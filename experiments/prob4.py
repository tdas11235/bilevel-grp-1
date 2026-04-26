import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupPDHG


nx, nz = 4, 1
groups = [[0, 1], [2, 3]]
c = np.array([1.0, 1.0, 0.0, 0.0])
mu = 1.0

# rotational stress problem
class Prob4(BilevelProblem):
    def __init__(self, mu=0.1): # Lower mu so x wants to move
        nx, nz = 4, 1
        self.groups = [[0, 1], [2, 3]]
        # We want to minimize x0 and x1
        self.c = c
        super().__init__(z_min=np.array([-3.14]), z_max=np.array([3.14]),
                         nx=nx, c=self.c, m=2, l=0)

    def A_sym(self, z):
        c, s = ca.cos(z[0]), ca.sin(z[0])
        return ca.vertcat(
            ca.horzcat(-c, s, -1, 0),
            ca.horzcat(-s, -c, 0, -1)
        )

    def b_sym(self, z):
        return ca.vertcat(-5.0, -5.0)
    
    def P_sym(self, z):
        return ca.MX(0, self.nx)

    def r_sym(self, z):
        return ca.MX(0, 1)
    

pdhg = GroupPDHG(
    c=c,
    groups=groups,
    mu=mu,
    lb_x=-50.0 * np.ones(nx),
    ub_x=50.0 * np.ones(nx),
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

z0 = np.array([-1.0])

sol = solver.solve(z0)
print(sol)
