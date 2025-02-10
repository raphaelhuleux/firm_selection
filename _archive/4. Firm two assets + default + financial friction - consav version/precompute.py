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
def get_coarse_exit_decision(par, sol):

    exit_policy = sol.exit_policy
    for iz in range(par.Nz):
        for ik in range(par.Nk):
            for ib in range(par.Nb):
                res = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.exit_policy_dense, par.z_grid[iz], par.b_grid[ib], par.k_grid[ik])
                if res > 0.0:
                    exit_policy[iz,ib,ik] = 1
                else:
                    exit_policy[iz,ib,ik] = 0

@nb.njit
def compute_exit_decision(par, sol, grid = 'dense'):
    """ 
    Iteratively update the exit decision function until convergence
    """
    if grid == 'dense':
        exit_policy_guess = np.zeros((par.Nz_dense,par.Nb_dense,par.Nk_dense)) 
    else:
        exit_policy_guess = np.zeros((par.Nz,par.Nb,par.Nk))

    error = 1 
    tol = 1e-4
    while error > tol:
        exit_policy_new = compute_exit_decision_step(exit_policy_guess, par, grid = grid)
        error = np.mean(np.abs(exit_policy_new - exit_policy_guess))
        print(error)
        exit_policy_guess = exit_policy_new

    if grid == "dense":
        sol.exit_policy_dense[...] = exit_policy_guess
    else:
        sol.exit_policy[...] = exit_policy_guess


@nb.njit(parallel=True)
def compute_exit_decision_step(exit_policy, par, grid = "dense"):
    """ 
    For a guess on the exit decision function, update the exit decision function by 
        - check if setting b_next = b_max yields positive profits
        - if not, check if an interior solution yields dividend profits (posible if diminishing b_next increases q by enough)
        - if not, the firm exits
    """

    if grid == 'dense':
        Nz, Nb, Nk = par.Nz_dense, par.Nb_dense, par.Nk_dense
        z_grid = par.z_grid_dense
        k_grid = par.k_grid_dense
        b_grid = par.b_grid_dense
    else:
        Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
        z_grid = par.z_grid
        k_grid = par.k_grid
        b_grid = par.b_grid

    div_max = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = z_grid[iz]
        for ik in range(Nk):
            k = k_grid[ik]
            for ib in range(Nb):
                b = b_grid[ib]
                b_max = b_grid[-1] # par.nu * (1-par.delta) * k
                k_next = (1-par.delta) * k_grid[ik]

                # Check div_policy when b_next = b_max 
                div_b_max = -objective_dividend_keeper(b_max, k_next, z, b, k, iz, exit_policy, par, grid)

                # If b_next = b_max not feasible, check for an interior solution
                if (div_b_max < 0):
                    b_next = optimizer(objective_dividend_keeper, b_grid[0], b_max, args=(k_next, z, b, k, iz, exit_policy, par, grid))
                    div_max[iz,ib,ik] = -objective_dividend_keeper(b_next, k_next, z, b, k, iz, exit_policy, par, grid)  
                else:
                    b_next = b_max 
                    div_max[iz,ib,ik] = div_b_max

    exit_policy_new = np.asarray(div_max < 0, dtype=np.float64)
    return exit_policy_new


@nb.njit
def objective_dividend_keeper(b_next, k_next, z, b, k, iz, exit_policy, par, grid):
    if grid == 'dense':
        P = par.P_dense
        z_grid = par.z_grid_dense
        k_grid = par.k_grid_dense
        b_grid = par.b_grid_dense
    else:
        P = par.P
        z_grid = par.z_grid
        k_grid = par.k_grid
        b_grid = par.b_grid

    q = debt_price_function(iz, k_next, b_next, par.r, exit_policy, P, k_grid, b_grid)
    div = z * k**par.alpha + q * b_next - b - par.cf  
    return -div


@nb.njit
def compute_q_matrix(par, sol, grid = "dense"):
    """ 
    Given an exit decision function, compute the price of debt for all choices of k_next, b_next, z 
    """
    q_mat = sol.q 

    if grid == 'dense':
        Nz, Nb, Nk = par.Nz_dense, par.Nb_dense, par.Nk_dense
        P = par.P_dense
        z_grid = par.z_grid_dense
        k_grid = par.k_grid_dense
        b_grid = par.b_grid_dense
        exit_policy = sol.exit_policy_dense

    else:
        Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
        P = par.P
        z_grid = par.z_grid
        k_grid = par.k_grid
        b_grid = par.b_grid
        exit_policy = sol.exit_policy

    for iz in range(Nz):
        for ik in range(Nk):
            for ib in range(Nb):
                k_next = k_grid[ik]
                b_next = b_grid[ib]
                q_mat[iz,ib,ik] = debt_price_function(iz, k_next, b_next, par.r, exit_policy, P, k_grid, b_grid)

@nb.njit
def objective_b_min_interp(b_next, z, b, k, par, sol):
    k_next = (1-par.delta) * k
    q = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.q, z, b_next, k_next)
    div = z * k**par.alpha + q * b_next - b - par.cf  
    return -div


@nb.njit 
def compute_b_min_interp(par, sol):
    
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
                b_max = par.b_grid[-1]

                if exit_policy[iz,ib,ik] == 1:
                    continue

                div_b_min = -objective_b_min_interp(b_grid[0], z, b, k, par, sol)
                
                if div_b_min < 0:
                    div_b_max = -objective_b_min_interp(b_max, z, b, k, par, sol)
                    if div_b_max < 0:
                        b_max = optimizer(objective_b_min_interp, b_grid[0], b_max, args=(z, b, k, par, sol))
                        div_b_max = -objective_b_min_interp(b_max, z, b, k, par, sol)

                    res = qe.optimize.root_finding.bisect(objective_b_min_interp, b_grid[0], b_max, args=(z, b, k, par, sol))
                    b_min = res.root
                    div_b_min = -objective_b_min_interp(b_min, z, b, k, par, sol)
                else:
                    b_min = b_grid[0]
                    
                b_min_keep[iz,ib,ik] = b_min


@nb.njit 
def compute_b_min(par, sol, grid = 'coarse'):
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

                div_b_min = -objective_dividend_keeper(b_grid[0], k_next, z, b, k, iz, exit_policy, par, grid)
                
                if div_b_min < 0:
                    div_b_max = -objective_dividend_keeper(b_max, k_next, z, b, k, iz, exit_policy, par, grid)
                    if div_b_max < 0:
                        b_max = optimizer(objective_dividend_keeper, b_grid[0], b_max, args=(k_next, z, b, k, iz, exit_policy, par, grid))
                        div_b_max = -objective_dividend_keeper(b_max, k_next, z, b, k, iz, exit_policy, par, grid)

                    res = qe.optimize.root_finding.bisect(objective_dividend_keeper, b_grid[0], b_max, args=(k_next, z, b, k, iz, exit_policy, par, grid))
                    b_min = res.root
                    div_b_min = -objective_dividend_keeper(b_min, k_next, z, b, k, iz, exit_policy, par, grid)
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
            q = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.q, z, b_next, k_next)
            #q = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
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


