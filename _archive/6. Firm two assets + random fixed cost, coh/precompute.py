import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from consav.linear_interp_3d import interp_3d
from numba import njit, prange
import quantecon as qe 
from consav.golden_section_search import optimizer 
from model_functions import * 

""" 
Compute exit decision
"""

@nb.njit 
def compute_exit_decision_adj(par, sol):
    b_grid = par.b_grid

    for iz in range(par.Nz):
        for ik in range(par.Nk):
            for im in range(par.Nm):
                if sol.exit_policy[iz,im,ik] == 1:
                    sol.exit_policy_adj[iz,im,ik] = 1
                else:
                    m = par.m_grid[im]
                    k = par.k_grid[ik]
                    z = par.z_grid[iz]

                    b_next = optimizer(objective_dividend_adj, b_grid[0], b_grid[-1], args=(iz, m, k, sol, par))
                    div = -objective_dividend_adj(b_next, iz, m, k, sol, par)  
                    if div >= 0.0:
                        sol.exit_policy_adj[iz,im,ik] = 0
                    else:
                        sol.exit_policy_adj[iz,im,ik] = 1


@nb.njit
def compute_exit_decision(par, sol):
    """ 
    Iteratively update the exit decision function until convergence
    """

    exit_policy = np.ones((par.Nz,par.Nm,par.Nk))

    error = 1 
    tol = 5e-4
    while error > tol:
        compute_q_matrix(exit_policy, par, sol)
        exit_policy_new = compute_exit_decision_step(sol, par)
        error = np.mean(np.abs(exit_policy_new - exit_policy))
        print(error)
        exit_policy = exit_policy_new

    sol.exit_policy[...] = exit_policy
    compute_q_matrix(sol.exit_policy, par, sol)



@nb.njit(parallel=True)
def compute_exit_decision_step(sol, par):
    """ 
    For a guess on the exit decision function, update the exit decision function by 
        - check if setting b_next = b_max yields positive profits
        - if not, check if an interior solution yields dividend profits (posible if diminishing b_next increases q by enough)
        - if not, the firm exits
    """

    Nz, Nm, Nk = par.Nz, par.Nm, par.Nk
    k_grid = par.k_grid
    m_grid = par.m_grid
    b_grid = par.b_grid

    div_max = np.zeros((Nz, Nm, Nk))

    for iz in prange(Nz):
        for ik in range(Nk):
            k = k_grid[ik]
            for im in range(Nm):
                m = m_grid[im]
                b_max = b_grid[-1] # par.nu * (1-par.delta) * k

                # Check div_policy when b_next = b_max 
                div_b_max = -objective_dividend_keeper(b_max, m, k, iz, sol, par)

                # If b_next = b_max not feasible, check for an interior solution
                if (div_b_max < 0):
                    b_next = optimizer(objective_dividend_keeper, b_grid[0], b_max, args=(m, k, iz, sol, par))
                    div_max[iz,im,ik] = -objective_dividend_keeper(b_next, m, k, iz, sol, par)  
                else:
                    b_next = b_max 
                    div_max[iz,im,ik] = div_b_max

    exit_policy_new = np.asarray(div_max <= 0, dtype=np.float64)
    return exit_policy_new

@nb.njit
def objective_dividend_keeper(b_next, m, k, iz, sol, par):
    k_next = (1-par.delta) * k
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m - k_next + q * b_next
    return -div

@nb.njit 
def objective_dividend_adj(b_next, iz, m, k, sol, par):
    k_next = (1-par.delta) * k + 1e-6 
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m - adj_cost - k_next + q * b_next
    return -div

@nb.njit
def compute_q_matrix(exit_policy, par, sol):
    """ 
    Given an exit decision function, compute the price of debt for all choices of k_next, b_next, z 
    """
    q_mat = sol.q 

    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    k_grid = par.k_grid
    b_grid = par.b_grid

    for iz in range(Nz):
        for ik in range(Nk):
            for ib in range(Nb):
                k_next = k_grid[ik]
                b_next = b_grid[ib]
                q_mat[iz,ib,ik] = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par)




