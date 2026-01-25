import numpy as np
from enum import Enum
from dataclasses import dataclass
from utils.group import PDHGStatus


class StepStatus:
    ACCEPTED = 0
    RESTART = 1
    TERMINATE = 2

@dataclass
class TerminatedPoint:
    z: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    nu: np.ndarray
    fval: float


class GAPTRSolver:
    def __init__(
            self,
            problem, pdhg_solver, retsoration_solver, *,
            eta=0.1, beta=0.5, tau=1e-4,
            eps=1e-6, kappa=0.1, eps_on=1e-6,
            max_iter=1000, t_min=1e-8,
            damp=0.7, amp=1.5, switch_ratio=0.7,
            verbose=False 
    ):
        """
        :param problem: BilevelProblem
        :param pdhg_solver: GroupPDHG
        :param retsoration_solver: RestorationNLP
        """
        self.problem = problem
        self.pdhg_solver = pdhg_solver
        self.restoration = retsoration_solver
        # algorithm params
        self.eta = eta
        self.beta = beta
        self.tau = tau
        self.eps = eps
        self.kappa = kappa
        self.eps_on = eps_on
        # internal global states
        self.restore_count = 0
        self.tol_mode = False
        self.min_g = np.inf
    
    def _project_box(self, z):
        return np.clip(z, self.problem.z_min, self.problem.z_max)
    
    def _effective_gradient(self, z, g):
        g_eff = np.where(
            
        )
