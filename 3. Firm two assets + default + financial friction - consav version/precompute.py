import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from consav.golden_section_search import optimizer 
from model_functions import * 

""" 
Compute exit decision
"""

@nb.njit
def compute_exit_decision(par, sol):
    """ 
    Iteratively update the exit decision function until convergence
    """
    exit_policy_guess = np.zeros((par.Nz,par.Nb,par.Nk)) 

    error = 1 
    tol = 1e-4
    while error > tol:
        exit_policy_new = compute_exit_decision_step(exit_policy_guess, par, sol)
        error = np.mean(np.abs(exit_policy_new - exit_policy_guess))
        print(error)
        exit_policy_guess = exit_policy_new

    sol.exit_policy[...] = exit_policy_guess


@nb.njit
def compute_exit_decision_step(exit_policy, par, sol):
    """ 
    For a guess on the exit decision function, update the exit decision function by 
        - check if setting b_next = b_max yields positive profits
        - if not, check if an interior solution yields dividend profits (posible if diminishing b_next increases q by enough)
        - if not, the firm exits
    """
    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    div_max = np.zeros((Nz, Nb, Nk))

    for iz in range(Nz):
        z = par.z_grid[iz]
        for ik in range(Nk):
            k = par.k_grid[ik]
            for ib in range(Nb):
                b = par.b_grid[ib]
                b_max = par.b_grid[-1] # par.nu * (1-par.delta) * k
                k_next = (1-par.delta) * par.k_grid[ik]

                # Check div_policy when b_next = b_max 
                div_b_max = -objective_dividend_keeper(b_max, k_next, z, b, k, iz, exit_policy, par)

                # If b_next = b_max not feasible, check for an interior solution
                if (div_b_max < 0):
                    b_next = optimizer(objective_dividend_keeper, par.b_grid[0], b_max, args=(k_next, z, b, k, iz, exit_policy, par))
                    div_max[iz,ib,ik] = -objective_dividend_keeper(b_next, k_next, z, b, k, iz, exit_policy, par)  
                else:
                    b_next = b_max 
                    div_max[iz,ib,ik] = div_b_max

    exit_policy_new = np.asarray(div_max < 0, dtype=np.float64)
    return exit_policy_new

@nb.njit
def objective_dividend_keeper(b_next, k_next, z, b, k, iz, exit_policy, par):
    q = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par.P, par.k_grid, par.b_grid)
    div = z * k**par.alpha + q * b_next - b - par.cf  
    return -div

@nb.njit
def compute_q_matrix(par, sol):
    """ 
    Given an exit decision function, compute the price of debt for all choices of k_next, b_next, z 
    """
    q_mat = sol.q 
    exit_policy = sol.exit_policy
    for iz in range(par.Nz):
        for ik in range(par.Nk):
            for ib in range(par.Nb):
                k_next = par.k_grid[ik]
                b_next = par.b_grid[ib]
                q_mat[iz,ib,ik] = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par.P, par.k_grid, par.b_grid)

@nb.njit 
def compute_b_min(par, sol):
    """ 
    Compute the minimum level of b_next that yields positive div_policy:
        - check div_policy when b_next = 0. If positive, stops
        - if negative, check for an iterior solution by using a root finder (b_next such that div_policy = 0)
        - to use the bisection, we need to find an upper bound on b_next that yields positive div_policy. We check first
        b_next = nu * k_next. If this is negative, we take an interior solution using an optimizer. 
    This is equivalent to solving the keeper problem!
    """
    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    z_grid, b_grid, k_grid = par.z_grid, par.b_grid, par.k_grid
    exit_policy = sol.exit_policy 
    delta = par.delta

    b_min_keep = sol.b_min_keep  

    for iz in range(Nz):
        z = z_grid[iz]
        for ik in range(Nk):
            k = k_grid[ik]
            for ib in range(Nb):
                b = b_grid[ib] 
                k_next = (1-delta) * k
                b_max = par.b_grid[-1]

                if exit_policy[iz,ib,ik] == 1:
                    continue

                div_b_min = -objective_dividend_keeper(b_grid[0], k_next, z, b, k, iz, exit_policy, par)
                
                if div_b_min < 0:
                    div_b_max = -objective_dividend_keeper(b_max, k_next, z, b, k, iz, exit_policy, par)
                    if div_b_max < 0:
                        b_max = optimizer(objective_dividend_keeper, b_grid[0], b_max, args=(k_next, z, b, k, iz, exit_policy, par))
                    res = qe.optimize.root_finding.bisect(objective_dividend_keeper, b_grid[0], b_max, args=(k_next, z, b, k, iz, exit_policy, par))
                    b_min = res.root
                else:
                    b_min = b_grid[0]
                b_min_keep[iz,ib,ik] = b_min

@nb.njit
def grid_search_k_max(z, b, k, iz, sol, par, Nb_next = 200, Nk_next = 200):

    k_choice = np.linspace((1-par.delta)*k, par.k_grid[-1], Nk_next)
    b_choice = np.linspace(b, par.b_grid[-1], Nb_next)

    for k_next in k_choice:    
        for b_next in b_choice:
            coh = z * k**par.alpha - b + (1-par.delta) * k
            q = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
            adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
            div = coh + b_next * q - adj_cost - par.cf - k_next 
            if div >= 0.0:
                k_max = k_next
                b_opt = b_next
                 
    return k_max, b_opt 

@nb.njit(parallel=True)
def compute_k_max(par, sol):

    k_max_adj = sol.k_max_adj

    for iz in nb.prange(par.Nz):
        z = par.z_grid[iz]
        for ik in range(par.Nk):
            k = par.k_grid[ik]
            for ib in range(par.Nb):
                b = par.b_grid[ib] 
                if sol.exit_policy[iz,ib,ik] == 1:
                    continue

                k_next, b_next = grid_search_k_max(z, b, k, iz, sol, par)

                k_max_adj[iz,ib,ik] = k_next


