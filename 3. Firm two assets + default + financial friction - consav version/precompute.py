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
    exit_policy_guess = np.zeros((par.N_z,par.N_b,par.N_k)) 

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
    N_z, N_b, N_k = par.N_z, par.N_b, par.N_k
    div_max = np.zeros((N_z, N_b, N_k))

    for iz in range(N_z):
        z = par.z_grid[iz]
        for ik in range(N_k):
            k = par.k_grid[ik]
            for ib in range(N_b):
                b = par.b_grid[ib]

                b_max = par.nu * (1-par.delta) * k
                k_next = (1-par.delta) * par.k_grid[ik]

                # Check dividends when b_next = b_max 
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
    for iz in range(par.N_z):
        for ik in range(par.N_k):
            for ib in range(par.N_b):
                k_next = par.k_grid[ik]
                b_next = par.b_grid[ib]
                q_mat[iz,ib,ik] = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par.P, par.k_grid, par.b_grid)


@nb.njit 
def compute_b_min(par, sol):
    """ 
    Compute the minimum level of b_next that yields positive dividends:
        - check dividends when b_next = 0. If positive, stops
        - if negative, check for an iterior solution by using a root finder (b_next such that dividends = 0)
        - to use the bisection, we need to find an upper bound on b_next that yields positive dividends. We check first
        b_next = nu * k_next. If this is negative, we take an interior solution using an optimizer. 

    """
    N_z, N_b, N_k = par.N_z, par.N_b, par.N_k
    z_grid, b_grid, k_grid = par.z_grid, par.b_grid, par.k_grid
    exit_policy = sol.exit_policy 
    delta = par.delta
    nu = par.nu 
    r = par.r
    b_min_keep = sol.b_min_keep  

    for iz in range(N_z):
        z = z_grid[iz]
        for ik in range(N_k):
            k = k_grid[ik]
            for ib in range(N_b):
                b = b_grid[ib] 
                k_next = (1-delta) * k
                b_max = nu * k_next

                if exit_policy[iz,ib,ik] == 1:
                    continue
                # objective_dividend_keeper(b_next, k_next, z, b, k, iz, exit_policy, par)
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
def grid_search_k_max(z, b, k, iz, sol, par, N_b_next = 100, N_k_next = 100):

    k_choice = np.linspace((1-par.delta)*k, par.k_grid[-1], N_k_next)

    for k_next in k_choice:
        b_choice = np.linspace(b, par.nu * k_next, N_b_next)
        for b_next in b_choice:
            coh = z * k**par.alpha - b + (1-par.delta) * k
            q = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
            adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
            div = coh + b_next * q - adj_cost - par.cf - k_next 
            if div >= 0.0:
                k_max = k_next
                b_opt = b_next
            if q == 0.0:
                break 

    return k_max, b_opt 

@nb.njit(parallel=True)
def compute_k_max(par, sol):

    k_max_adj = sol.k_max_adj

    for iz in nb.prange(par.N_z):
        z = par.z_grid[iz]
        for ik in range(par.N_k):
            k = par.k_grid[ik]
            for ib in range(par.N_b):
                b = par.b_grid[ib] 
                if sol.exit_policy[iz,ib,ik] == 1:
                    continue

                k_next, b_next = grid_search_k_max(z, b, k, iz, sol, par)

                k_max_adj[iz,ib,ik] = k_next

