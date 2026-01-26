import numpy as np


def update_active_groups(
    grad_x, groups, mu,
    active_prev=None, eps_off=1e-8,
):
    """
    grad_x : smooth gradient wrt x
    groups : list of index arrays
    mu     : group l1 parameter

    Returns:
        active : boolean array of len(groups)
        gnorms : array of group gradient norms
    """
    ng = len(groups)
    gnorms = np.zeros(ng)
    active = np.zeros(ng, dtype=bool)
    for i, G in enumerate(groups):
        gnorms[i] = np.linalg.norm(grad_x[G])
    if active_prev is None:
        active = gnorms >= mu
    else:
        active = active_prev.copy()
        active[gnorms >= mu] = True
        active[gnorms < mu - eps_off] = False
    return active, gnorms
