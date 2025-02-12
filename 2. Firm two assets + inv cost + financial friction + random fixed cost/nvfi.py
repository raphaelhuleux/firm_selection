import numpy as np 
from consav.linear_interp_2d import interp_2d
from consav.linear_interp_3d import interp_3d

from numba import njit, prange
from model_functions import * 
from precompute import objective_dividend_keeper
import quantecon as qe 
from consav.golden_section_search import optimizer 

""" 
Problem with the code: 
when solving the adjuster problem, some intermediate values of k_next between k_min and k_max are not feasible,
even if k_min and k_max are feasible! this means that there is some strong non-monotonicity in the problem: 
increasing k_next might actually make you more.
Maybe the issue is with the price function q that looks like a step function
Also try to redo compute_exit_decision_step with a dense grid_search?
"""

"""
VFI 
"""

""" 
Solve keeper problem 
"""

@njit 
def bellman_keep(b_next, b, k, iz, q_mat, W, par):
    z = par.z_grid[iz]
    coh = k**par.alpha * z - b 
    k_next = (1-par.delta) * k
    q = interp_2d(par.b_grid, par.k_grid, q_mat[iz], b_next, k_next)
    div = coh + b_next * q 

    penalty = 0.0 
    if div < 0.0:
        penalty = np.abs(div*1e6)
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return -V + penalty

@njit(parallel = True)
def solve_keep_analytical(W, exit_policy, q_mat, b_min_keep, par):

    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy = np.zeros((Nz, Nb, Nk))
    b_policy = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk): 
                if exit_policy[iz,ib,ik]:
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                else:
                    k = par.k_grid[ik]
                    k_next = (1-par.delta) * k
                    b_next =  b_min_keep[iz,ib,ik]

                    k_policy[iz,ib,ik] = k_next
                    b_policy[iz,ib,ik] = b_next
                    V_new[iz,ib,ik] = -bellman_keep(b_next, b, k, iz, q_mat, W, par)

    return V_new, k_policy, b_policy

@njit(parallel = True)
def solve_keep(W, exit_policy, q_mat, b_min_keep, par):

    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy = np.zeros((Nz, Nb, Nk))
    b_policy = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk): 
                if exit_policy[iz,ib,ik]:
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                else:
                    k = par.k_grid[ik]
                    k_next = (1-par.delta) * k
                    b_min = b_min_keep[iz,ib,ik]
                    b_next = optimizer(bellman_keep, b_min, par.b_grid[-1], args = (b, k, iz, q_mat, W, par))

                    k_policy[iz,ib,ik] = k_next
                    b_policy[iz,ib,ik] = b_next
                    V_new[iz,ib,ik] = -bellman_keep(b_next, b, k, iz, q_mat, W, par)

    return V_new, k_policy, b_policy

""" 
Solve adjuster problem
"""

@njit 
def bellman_adj(k_next, b, k, iz, q_mat, b_keep, W, par):
    z = par.z_grid[iz]

    y = z * k**par.alpha 
    y_new = z * (k_next / (1-par.delta))**par.alpha

    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    b_tilde = b + adj_cost + (y_new - y) + k_next - (1-par.delta) * k
    b_next = interp_2d(par.b_grid, par.k_grid, b_keep[iz], b_tilde, k_next / (1-par.delta))
    q = interp_2d(par.b_grid, par.k_grid, q_mat[iz], b_next, k_next)

    div = y + (1-par.delta) * k - b - k_next + b_next * q - adj_cost

    penalty = 0.0
    if div < 0:
        penalty += np.abs(div*1e6)
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)

    #V = bellman_keep(b_next, b_tilde, k_next / (1-par.delta), iz, W, par, sol)
    
    return -V + penalty

@njit 
def bellman_invest(k_next, b_next, b, k, iz, q_mat, W, par): 

    z = par.z_grid[iz]

    coh = z * k**par.alpha + (1-par.delta) * k - b 
    q = interp_2d(par.b_grid, par.k_grid, q_mat[iz], b_next, k_next)
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next * q    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V 


@njit(parallel = True)
def solve_adj(W, exit_policy_adj, q_mat, k_max_adj, b_keep, par):
    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy = np.zeros((Nz, Nb, Nk))
    b_policy = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = par.z_grid[iz]
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk):
                k = par.k_grid[ik]

                if exit_policy_adj[iz,ib,ik]:
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                else:
                    k_min = (1-par.delta) * k + 1e-6
                    k_max = k_max_adj[iz,ib,ik]
                    k_next = optimizer(bellman_adj, k_min, k_max, args = (b, k, iz, q_mat, b_keep, W, par))

                    y = z * k**par.alpha 
                    y_new = z * (k_next / (1-par.delta))**par.alpha
                    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
                    b_tilde = b + adj_cost + (y_new - y) + k_next - (1-par.delta) * k
                    b_next = interp_2d(par.b_grid, par.k_grid, b_keep[iz], b_tilde, k_next / (1-par.delta))

                    b_policy[iz,ib,ik] = b_next
                    k_policy[iz,ib,ik] = k_next
                    V_new[iz,ib,ik] = bellman_invest(k_next, b_next, b, k, iz, q_mat, W, par)

    return V_new, k_policy, b_policy

""" 
NVFI
"""

def solve_problem_firm_trans(trans, ss, par):

    V, k_policy, b_policy = (np.zeros((par.T, par.Nz, par.Nb, par.Nk)) for _ in range(3))
    q_mat = trans.q 
    exit_policy = trans.exit_policy
    exit_policy_adj = trans.exit_policy_adj
    b_min_keep = ss.b_min_keep
    k_max_adj = ss.k_max_adj

    for t in reversed(range(par.T)):
        if t == par.T - 1:
            Vtemp = ss.V
        else:
            Vtemp = V[t+1]
            
        V[t], k_policy[t], b_policy[t] = nvfi_step(Vtemp, q_mat[t], exit_policy[t], exit_policy_adj[t], b_min_keep[t], k_max_adj[t], par, solve_b = 'optimizer')
    
    trans.V[...] = V
    trans.k_policy[...] = k_policy
    trans.b_policy[...] = b_policy


def nvfi_step(V, q_mat, exit_policy, exit_policy_adj, b_min_keep, k_max_adj, par, solve_b = 'analytical'):
    W = par.beta * fast_expectation(par.P, V)
    W = compute_expectation_omega(W, par)
    if solve_b == 'analytical':
        V_keep, k_keep, b_keep = solve_keep_analytical(W, exit_policy, q_mat, b_min_keep, par)
    else:
        V_keep, k_keep, b_keep = solve_keep(W, exit_policy, q_mat, b_min_keep, par)
    V_adj, k_adj, b_adj = solve_adj(W, exit_policy_adj, q_mat, k_max_adj, b_keep, par)

    k_policy = np.where(V_keep >= V_adj, k_keep, k_adj)
    b_policy = np.where(V_keep >= V_adj, b_keep, b_adj)
    V_new = np.maximum(V_keep, V_adj)

    return V_new, k_policy, b_policy

def solve_nvfi_ss(ss, par, tol = 1e-4):
    error = 1

    V_init = np.zeros((par.Nz, par.Nb, par.Nk))
    V = V_init.copy()
    n = 0 

    while error > tol:
        Vnew, k_policy, b_policy = nvfi_step(V, ss.q, ss.exit_policy, ss.exit_policy_adj, ss.b_min_keep, ss.k_max_adj, par, solve_b = par.solve_b)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if par.howard & (n > 5):
            V = howard_nvfi(V, k_policy, b_policy, ss, par)
        n += 1

    ss.inaction[...] = k_policy == (1-par.delta) * par.k_grid[None, None, :]
    ss.k_policy[...] = k_policy
    ss.b_policy[...] = b_policy
    ss.V[...] = V
    compute_optimal_div_policy(b_policy, k_policy, par, ss)

""" 
Howard
"""

@njit(parallel = True)
def howard_step_nvfi(W, k_policy, b_policy, ss, par):
    Nz, Nb, Nk = W.shape
    V_new = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk):                 
                if ss.exit_policy[iz,ib,ik]:
                    V_new[iz, ib, ik] = 0

                else:
                    k = par.k_grid[ik]
                    k_next = k_policy[iz, ib, ik]
                    b_next = b_policy[iz, ib, ik]

                    if k_next == (1-par.delta) * k:
                        V_new[iz, ib, ik] = -bellman_keep(b_next, b, k, iz, ss.q, W, par)
                    else:
                        V_new[iz, ib, ik] = bellman_invest(k_next, b_next, b, k, iz, ss.q, W, par)
    return V_new

def howard_nvfi(V, k_policy, b_policy, ss,  par, tol = 1e-4, iter_max = 1000):

    for n in range(20):
        W = par.beta * fast_expectation(par.P, V)
        W = compute_expectation_omega(W, par)
        V = howard_step_nvfi(W, k_policy, b_policy, ss, par)

    return V 