import casadi as ca
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import GAPTRSolver
from utils.restoration import RestorationNLP
from utils.bilevel import BilevelProblem
from utils.group import GroupPDHG

class Prob1(BilevelProblem):
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        nz = 2
        nx = 10
        m = 6
        l = 2
        z_min = -2 * np.ones(nz)
        z_max = 2 * np.ones(nz)
        self.c = np.ones(nx)
        # inequality data
        self.A0 = rng.standard_normal((m, nx))
        self.A1 = 0.2 * rng.standard_normal((m, nx))
        self.A2 = 0.2 * rng.standard_normal((m, nx))
        self.b0 = 3.0 * np.ones(m)
        # equality data
        self.P0 = rng.standard_normal((l, nx))
        self.r0 = np.zeros(l)
        self.v = rng.standard_normal(l)
        super().__init__(z_min, z_max, nx, self.c, m, l)

    def A_sym(self, z):
        return (
            ca.DM(self.A0)
            + z[0] * ca.DM(self.A1)
            + z[1] * ca.DM(self.A2)
        )

    def b_sym(self, z):
        return ca.DM(self.b0) + 0.2 * z[0] * ca.DM.ones(self.m)

    def P_sym(self, z):
        return ca.DM(self.P0)

    def r_sym(self, z):
        return ca.DM(self.r0) + z[1] * ca.DM(self.v)


groups = [
    np.array([0, 1, 2]),
    np.array([2, 3, 4]),
    np.array([4, 5, 6]),
    np.array([6, 7, 8, 9]),
]

mu = 0.5

pdhg = GroupPDHG(
    c=np.ones(10),
    groups=groups,
    mu=mu,
    lb_x=-5.0 * np.ones(10),
    ub_x=5.0 * np.ones(10),
    tau=5e-2,
    sigma=5e-2,
    theta=0.5,
    max_iter=10000,
    tol=1e-2,
    tol_cons=1e-4,
    verbose=False
)

problem = Prob1()
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
    eps=5e-3,
    eps_off=1e-6,
    max_iter=200,
    amp=1.2,
    damp=0.9,
    verbose=False
)

z0 = np.zeros(problem.nz)

sol = solver.solve(z0)
print(sol)