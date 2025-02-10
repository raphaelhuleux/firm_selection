import numpy as np 
from consav.linear_interp_2d import interp_2d, interp_2d_vec
from consav.linear_interp_3d import interp_3d

import numba as nb
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

@nb.njit 
def bellman_keep(b_next, m, k, iz, W, par, sol):

    k_next = (1-par.delta) * k
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m + b_next * q - k_next 

    penalty = 0.0 
    if div < 0.0:
        penalty = div**2 * 1e5
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return -V + penalty

@nb.njit(parallel = True)
def solve_keep(W, par, sol):

    V_new = np.zeros((par.Nz, par.Nm, par.Nk))
    k_policy = np.zeros((par.Nz, par.Nm, par.Nk))
    b_policy = np.zeros((par.Nz, par.Nm, par.Nk))

    for iz in nb.prange(par.Nz):
        for im in range(par.Nm):
            m = par.m_grid[im]
            for ik in range(par.Nk): 
                if sol.exit_policy[iz,im,ik]:
                    V_new[iz,im,ik] = 0
                    k_policy[iz,im,ik] = 0
                    b_policy[iz,im,ik] = 0
                else:
                    k = par.k_grid[ik]
                    k_next = (1-par.delta) * k
                    b_next =  optimizer(bellman_keep, par.b_grid[0], par.b_grid[-1], args = (m, k, iz, W, par, sol))

                    k_policy[iz,im,ik] = k_next
                    b_policy[iz,im,ik] = b_next
                    V_new[iz,im,ik] = -bellman_keep(b_next, m, k, iz, W, par, sol)

    return V_new, k_policy, b_policy


""" 
Solve adjuster problem
"""

@nb.njit 
def bellman_adj(k_next, m, k, iz, par, sol, W, b_policy_keep):

    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 

    m_adj = m - adj_cost
    b_next = interp_2d(par.m_grid, par.k_grid, b_policy_keep[iz], m_adj, k_next / (1-par.delta))
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m - adj_cost - k_next + b_next * q 

    penalty = 0.0
    if div < 0:
        penalty += div**2*1e5
    
    V = bellman_keep(b_next, m_adj, k_next / (1-par.delta), iz, W, par, sol)
    #V = interp_2d(par.m_grid, par.k_grid, V_keep[iz], m_adj, k_next)
    
    return V

@nb.njit(parallel = True)
def solve_adj(W, b_policy_keep, par, sol):

    V_new = np.zeros((par.Nz, par.Nm, par.Nk))
    k_policy = np.zeros((par.Nz, par.Nm, par.Nk))
    b_policy = np.zeros((par.Nz, par.Nm, par.Nk))

    for iz in nb.prange(par.Nz):
        for im in range(par.Nm):
            m = par.m_grid[im]
            for ik in range(par.Nk):
                k = par.k_grid[ik]

                if sol.exit_policy_adj[iz,im,ik]:
                    V_new[iz,im,ik] = 0
                    k_policy[iz,im,ik] = 0
                    b_policy[iz,im,ik] = 0
                else:
                    k_min = (1-par.delta) * k + 1e-6
                    k_max = par.k_grid[-1] #sol.k_max_adj[iz,ib,ik]
                    k_next = optimizer(bellman_adj, k_min, k_max, args = (m, k, iz, par, sol, W, b_policy_keep))
                    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k
                    m_adj = m - adj_cost 

                    b_next = interp_2d(par.m_grid, par.k_grid, b_policy_keep[iz], m_adj, k_next)
                    b_policy[iz,im,ik] = b_next
                    k_policy[iz,im,ik] = k_next 
                    V_new[iz,im,ik] = bellman_howard(k_next, b_next, m, k, iz, par, sol, W)

    return V_new, k_policy, b_policy

""" 
NVFI
"""

def nvfi_step_analytical(V, par, sol):
    W = par.beta * compute_expectation(V, par)
    V_keep, k_policy_keep, b_policy_keep = solve_keep(W, par, sol)
    V_adj, k_policy_adj, b_policy_adj = solve_adj(W, b_policy_keep, par, sol)

    k_policy = np.where(V_keep >= V_adj, k_policy_keep, k_policy_adj)
    b_policy = np.where(V_keep >= V_adj, b_policy_keep, b_policy_adj)
    V_new = np.maximum(V_keep, V_adj)

    return V_new, k_policy, b_policy

def solve_nvfi_analytical(par, sol, tol = 1e-4, do_howard = True):
    error = 1

    V_init = np.zeros((par.Nz, par.Nm, par.Nk))
    V = V_init.copy()

    while error > tol:
        Vnew, k_policy, b_policy = nvfi_step_analytical(V, par, sol)
        error = np.mean(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard_nvfi(V, k_policy, b_policy, par, sol)

    #sol.inaction[...] = k_policy == (1-par.delta) * par.k_grid[None, None, :]
    sol.k_policy[...] = k_policy
    sol.b_policy[...] = b_policy
    sol.V[...] = V
    compute_optimal_div_policy(b_policy, k_policy, par, sol)

""" 
Howard
"""

@nb.njit 
def bellman_howard(k_next, b_next, m, k, iz, par, sol, W): 

    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = m - adj_cost - k_next + b_next * q    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V 


@nb.njit(parallel = True)
def howard_step_nvfi(W, k_policy, b_policy, par, sol):
    V_new = np.zeros((par.Nz, par.Nm, par.Nk))

    for iz in nb.prange(par.Nz):
        for im in range(par.Nm):
            m = par.m_grid[im]
            for ik in range(par.Nk):                 
                if sol.exit_policy[iz,im,ik]:
                    V_new[iz, im, ik] = 0
                else:
                    k = par.k_grid[ik]
                    k_next = k_policy[iz, im, ik]
                    b_next = b_policy[iz, im, ik]

                    if k_next == (1-par.delta) * k:
                        V_new[iz, im, ik] = -bellman_keep(b_next, m, k, iz, W, par, sol)
                    else:
                        V_new[iz, im, ik] = bellman_howard(k_next, b_next, m, k, iz, par, sol, W)
    return V_new

def howard_nvfi(V, k_policy, b_policy, par, sol):

    for n in range(30):
        W = par.beta * compute_expectation(V, par)
        V = howard_step_nvfi(W, k_policy, b_policy, par, sol)

    return V 