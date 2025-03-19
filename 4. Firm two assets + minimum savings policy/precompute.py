import numpy as np 
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
from consav.golden_section_search import optimizer 
from model_functions import * 

""" 
Compute exit decision
"""

def compute_exit_decision_trans(r_trans, ss, par):

    q_trans = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
    exit_policy_trans = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
    exit_policy_adj_trans = np.zeros((par.T, par.Nz, par.Nb, par.Nk))

    for t in reversed(range(par.T)):
        print(t)
        if t == par.T-1:
            q_trans[t] = compute_q_matrix(ss.exit_policy, ss.r, par)
            exit_policy_trans[t] = compute_exit_decision_trans_step(ss.exit_policy, q_trans[t], par)

        else:
            q_trans[t] = compute_q_matrix(exit_policy_trans[t+1], r_trans[t+1], par)
            exit_policy_trans[t] = compute_exit_decision_trans_step(exit_policy_trans[t+1], q_trans[t], par)

    return exit_policy_trans, exit_policy_adj_trans, q_trans

@njit
def compute_exit_decision_trans_step(exit_policy, q, par, tol = 5e-4):

    error = 1 
    while error > tol:
        exit_policy_new = compute_exit_decision_step(q, par)
        error = np.sum(np.abs(exit_policy_new - exit_policy))
        exit_policy = exit_policy_new
    #exit_policy[:,:,0] = 1

    return exit_policy
  
def compute_exit_decision_ss(r, par):
    """ 
    Iteratively update the exit decision function until convergence
    """

    exit_policy = np.zeros((par.Nz,par.Nb,par.Nk))
    #exit_policy[:,:,0] = 1

    error = 1 
    tol = 5e-4
    while error > tol:
        q = compute_q_matrix(exit_policy, r, par)
        exit_policy_new = compute_exit_decision_step(q, par)
        error = np.sum(np.abs(exit_policy_new - exit_policy))
        print(error)
        exit_policy = exit_policy_new
    

    return exit_policy, q 


@njit(parallel=True)
def compute_exit_decision_step(q, par):
    """ 
    For a given price function, compute the maximum potential dividends when not adjusting k, if div negative, exit.
    """

    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    z_grid = par.z_grid
    k_grid = par.k_grid
    b_grid = par.b_grid

    div_max = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = z_grid[iz]
        for ik in range(Nk):
            k = k_grid[ik]
            k_next = (1-par.delta) * k

            for ib in range(Nb):
                b = b_grid[ib]
                b_max = np.minimum(par.nu*k_next, par.b_grid[-1])
            
                # Check div_policy when b_next = b_max 
                div_b_max = -objective_dividend_keeper(b_max, k_next, z, b, k, iz, q, par)

                # If b_next = b_max not feasible, check for an interior solution
                if (div_b_max < 0):
                    b_next = optimizer(objective_dividend_keeper, b_grid[0], b_max, args=(k_next, z, b, k, iz, q, par))
                    div_max[iz,ib,ik] = -objective_dividend_keeper(b_next, k_next, z, b, k, iz, q, par)  
                else:
                    b_next = b_max 
                    div_max[iz,ib,ik] = div_b_max

    exit_policy_new = np.asarray(div_max <= 0, dtype=np.float64)
    return exit_policy_new

@njit 
def objective_dividend_adj(b_next, iz, b, k, q_mat, par):
    z = par.z_grid[iz]
    k_next = (1-par.delta) * k + 1e-6 
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
    q = interp_2d(par.b_grid, par.k_grid, q_mat[iz], b_next, k_next)
    div = z * k**par.alpha + q * b_next - b + (1-par.delta) * k - adj_cost - k_next
    return -div

@njit
def objective_dividend_keeper(b_next, k_next, z, b, k, iz, q_mat, par):
    q = interp_2d(par.b_grid, par.k_grid, q_mat[iz], b_next, k_next)
    div = z * k**par.alpha + q * b_next - b
    return -div

@njit(parallel=True)
def compute_q_matrix(exit_policy, r, par):
    """ 
    Given an exit decision function, compute the price of debt for all choices of k_next, b_next, z 
    """

    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    k_grid = par.k_grid
    b_grid = par.b_grid

    q_mat = np.zeros((Nz, Nb, Nk)) 

    for iz in prange(Nz):
        for ik in range(Nk):
            for ib in range(Nb):
                k_next = k_grid[ik]
                b_next = b_grid[ib]
                q_mat[iz,ib,ik] = debt_price_function(iz, k_next, b_next, r, exit_policy, par)

    return q_mat


@njit
def grid_search_k_max(z, b, k, iz, q_mat, par, Nb_next = 150, Nk_next = 150):

    k_choice = np.linspace((1-par.delta)*k, par.k_grid[-1], Nk_next)
    k_max = -np.inf
    for k_next in k_choice:    
        b_choice = np.linspace(0, par.nu * k_next, Nb_next)
        for b_next in b_choice:
            coh = z * k**par.alpha - b + (1-par.delta) * k
            q = interp_2d(par.b_grid, par.k_grid, q_mat[iz], b_next, k_next)
            adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
            div = coh + b_next * q - adj_cost - k_next 

            if div >= 0.0:
                k_max = k_next
                b_opt = b_next
                 
    return k_max, b_opt 

@njit(parallel=True)
def compute_k_max(q_mat, exit_policy, par):

    k_max_adj = np.zeros((par.Nz, par.Nb, par.Nk))
    exit_policy_adj = np.zeros_like(exit_policy)

    for iz in prange(par.Nz):
        z = par.z_grid[iz]
        for ik in range(par.Nk):
            k = par.k_grid[ik]
            for ib in range(par.Nb):
                b = par.b_grid[ib] 
                if exit_policy[iz,ib,ik] == 1:
                    exit_policy_adj[iz,ib,ik] = 1
                else:
                    k_next, _ = grid_search_k_max(z, b, k, iz, q_mat, par)
                    if k_next == -np.inf:
                        exit_policy_adj[iz,ib,ik] = 1
                    else:
                        exit_policy_adj[iz,ib,ik] = 0
                        k_max_adj[iz,ib,ik] = k_next #par.k_grid[-1]
    
    return k_max_adj, exit_policy_adj

