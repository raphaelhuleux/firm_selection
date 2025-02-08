import numpy as np 
from consav.linear_interp_2d import interp_2d
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

@njit 
def bellman_keep(b_next, m, k, iz, W, par, sol):

    k_next = (1-par.delta) * k
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m + b_next * q - k_next 

    penalty = 0.0 
    if div < 0.0:
        penalty = np.abs(div*1e6)
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return -V + penalty

@njit(parallel = True)
def solve_keep(W, par, sol):

    Nz, Nm, Nk = W.shape

    V_new = np.zeros((Nz, Nm, Nk))
    k_policy = np.zeros((Nz, Nm, Nk))
    b_policy = np.zeros((Nz, Nm, Nk))

    for iz in prange(Nz):
        for im in range(Nm):
            m = par.m_grid[im]
            for ik in range(Nk): 
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

@nb.njit 
def compute_expectation(V, par):
    W = np.zeros((par.Nz, par.Nb, par.Nk))
    for iz in range(par.Nz):
        for ik_next in range(par.Nk):
            for ib_next in range(par.Nb):
                V_temp = 0
                for iz_next in range(par.Nz):
                    for iomega in range(par.Nomega):
                        P = par.P[iz,iz_next] * par.omega_p[iomega]
                        k_next = par.k_grid[ik_next]
                        b_next = par.b_grid[ib_next]
                        z_next = par.z_grid[iz_next]
                        omega = par.omega_grid[iomega]
                        m_next = z_next * k_next**par.alpha + (1-par.delta) * k_next - b_next - omega 
                        V_temp += P * interp_2d(par.m_grid, par.k_grid, V[iz_next], m_next, k_next)

                W[iz,ib_next,ik_next] = V_temp

    return W


""" 
Solve adjuster problem
"""

@njit 
def bellman_adj(k_next, m, k, iz, par, sol, V_keep, b_policy_keep):

    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 

    m_adj = m - adj_cost
    b_next = interp_2d(par.m_grid, par.k_grid, b_policy_keep[iz], m_adj, k_next / (1-par.delta))
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m - adj_cost - k_next + b_next * q 

    penalty = 0.0
    if div < 0:
        penalty += np.abs(div*1e3)
    
    V = interp_2d(par.m_grid, par.k_grid, V_keep[iz], m_adj, k_next / (1-par.delta))
    
    return -V + penalty

@njit
def solve_adj(V_keep, b_policy_keep, par, sol):
    Nz, Nm, Nk = V_keep.shape

    V_new = np.zeros((Nz, Nm, Nk))
    k_policy = np.zeros((Nz, Nm, Nk))
    b_policy = np.zeros((Nz, Nm, Nk))

    for iz in range(Nz):
        for im in range(Nm):
            m = par.m_grid[im]
            for ik in range(Nk):
                k = par.k_grid[ik]

                if sol.exit_policy_adj[iz,im,ik]:
                    V_new[iz,im,ik] = 0
                    k_policy[iz,im,ik] = 0
                    b_policy[iz,im,ik] = 0
                else:
                    k_min = (1-par.delta) * k + 1e-6
                    k_max = par.k_grid[-1] #sol.k_max_adj[iz,ib,ik]
                    k_next = optimizer(bellman_adj, k_min, k_max, args = (m, k, iz, par, sol, V_keep, b_policy_keep))
                    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k
                    m_adj = m - adj_cost 

                    b_next = interp_2d(par.m_grid, par.k_grid, b_policy_keep[iz], m_adj, k_next / (1-par.delta))
                    b_policy[iz,im,ik] = b_next
                    k_policy[iz,im,ik] = k_next 
                    V_new[iz,im,ik] = -bellman_adj(k_next, m, k, iz, par, sol, V_keep, b_policy_keep)

    return V_new, k_policy, b_policy

""" 
NVFI
"""

def nvfi_step_analytical(V, par, sol):
    W = par.beta * compute_expectation(V, par)
    V_keep, k_policy_keep, b_policy_keep = solve_keep(W, par, sol)
    V_adj, k_policy_adj, b_policy_adj = solve_adj(V_keep, b_policy_keep, par, sol)

    k_policy = np.where(V_keep >= V_adj, k_policy_keep, k_policy_adj)
    b_policy = np.where(V_keep >= V_adj, b_policy_keep, b_policy_adj)
    V_new = np.maximum(V_keep, V_adj)

    return V_new, k_policy, b_policy

def solve_nvfi_analytical(par, sol, tol = 1e-4, do_howard = True):
    error = 1

    V_init = np.zeros((par.Nz, par.Nb, par.Nk))
    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = nvfi_step_analytical(V, par, sol)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard_nvfi(V, k_policy, b_policy, par, sol)


    sol.inaction[...] = k_policy == (1-par.delta) * par.k_grid[None, None, :]
    sol.k_policy[...] = k_policy
    sol.b_policy[...] = b_policy
    sol.V[...] = V
    compute_optimal_div_policy(b_policy, k_policy, par, sol)

""" 
Howard
"""

@njit 
def bellman_howard(k_next, b_next, m, k, iz, par, sol, W): 

    z = par.z_grid[iz]

    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = m - adj_cost - k_next + b_next * q    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V 


@njit(parallel = True)
def howard_step_nvfi(W, k_policy, b_policy, par, sol):
    Nz, Nb, Nk = W.shape
    V_new = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = par.z_grid[iz]
        for im in range(Nb):
            m = par.m_grid[im]
            for ik in range(Nk):                 
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

def howard_nvfi(V, k_policy, b_policy, par, sol, tol = 1e-4, iter_max = 1000):

    for n in range(50):
        W = par.beta * compute_expectation(V, par)
        V = howard_step_nvfi(W, k_policy, b_policy, par, sol)

    return V 