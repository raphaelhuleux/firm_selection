import numpy as np 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
from model_functions import * 

"""
VFI 
"""

@njit 
def bellman_adj(b_next, k_next, m, k, iz, par, sol, W):

    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = m - adj_cost - k_next + b_next * q 

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V

@njit 
def bellman_keep(b_next, m, k, iz, par, sol, W):
    k_next = (1-par.delta) * k
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = m + q * b_next - k_next 

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V


@njit
def grid_search_adj(m, k, iz, k_max, W, par, sol, Nb_choice = 100, Nk_choice = 100):

    Vmax = -np.inf 

    k_min = (1-par.delta) * k + 1e-8
    k_choice = np.linspace(k_min, k_max, Nk_choice)
    b_choice = np.linspace(par.b_grid[0], par.b_grid[-1], Nb_choice)

    for k_next in k_choice:
        for b_next in b_choice:
            V = bellman_adj(b_next, k_next, m, k, iz, par, sol, W)
            if V > Vmax:
                Vmax = V
                b_opt = b_next
                k_opt = k_next

    return Vmax, b_opt, k_opt

@njit
def grid_search_keep(m, k, iz, b_min, W, par, sol, Nb_choice = 100):

    Vmax = -np.inf 
    b_choice = np.linspace(b_min, par.b_grid[-1], Nb_choice)
    b_opt = 0.0
    
    for b_next in b_choice:
        V = bellman_keep(b_next, m, k, iz, par, sol, W)

        if V > Vmax:
            Vmax = V
            b_opt = b_next
    
    return Vmax, b_opt


@njit(parallel = True)
def vfi_step(V, par, sol):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    b_policy = np.empty_like(V)


    W = par.beta * compute_expectation(V, par)

    for iz in prange(par.Nz):
        for ik in range(par.Nk):
            for im in range(par.Nm):

                if sol.exit_policy[iz,im,ik]:
                    V_new[iz, im, ik] = 0.0
                    k_policy[iz, im, ik] = 0.0
                    b_policy[iz, im, ik] = 0.0
                else:
                    m = par.m_grid[im]
                    k = par.k_grid[ik]

                    k_max = par.k_grid[-1] # sol.k_max_adj[iz, im, ik]
                    Vinv, b_inv, k_inv = grid_search_adj(m, k, iz, k_max, W, par, sol, Nb_choice = par.Nb_choice, Nk_choice = par.Nk_choice)

                    b_min_keep = par.b_grid[0] # sol.b_min_keep[iz, im, ik]
                    Vina, b_ina = grid_search_keep(m, k, iz, b_min_keep, W, par, sol, Nb_choice = par.Nb_choice)

                    if Vinv > Vina:
                        V_new[iz, im, ik] = Vinv
                        k_policy[iz, im, ik] = k_inv
                        b_policy[iz, im, ik] = b_inv
                    else:   
                        V_new[iz, im, ik] = Vina
                        k_policy[iz, im, ik] = (1-par.delta) * k
                        b_policy[iz, im, ik] = b_ina

              
    return V_new, k_policy, b_policy

def solve_vfi_grid_search(par, sol, tol = 1e-4, do_howard = True):
    error = 1

    V_init = np.zeros((par.Nz, par.Nm, par.Nk))
    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = vfi_step(V, par, sol)
        error = np.mean(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard(V, k_policy, b_policy, par, sol)

    #sol.inaction[...] = k_policy > (1-par.delta) * par.k_grid[None, None, :]
    sol.k_policy[...] = k_policy
    sol.b_policy[...] = b_policy
    sol.V[...] = V
    compute_optimal_div_policy(b_policy, k_policy, par, sol)


@njit(parallel = True)
def howard_step(W, k_policy, b_policy, par, sol):

    V_new = np.empty_like(W)

    for iz in prange(par.Nz):
        for ik in range(par.Nk):
            for im in range(par.Nm):
                if sol.exit_policy[iz,im,ik]:
                    V_new[iz, im, ik] = 0.0
                else:
                    m = par.m_grid[im]
                    k = par.k_grid[ik]
                    b_next = b_policy[iz, im, ik]
                    k_next = k_policy[iz, im, ik]
                    
                    if k_next == (1-par.delta) * k:
                        V_new[iz, im, ik] = bellman_keep(b_next, m, k, iz, par, sol, W)
                    else:
                        V_new[iz, im, ik] = bellman_adj(b_next, k_next, m, k, iz, par, sol, W)
                    
    return V_new

def howard(V, k_policy, b_policy, par, sol):

    for n in range(50):
        W = par.beta * compute_expectation(V, par)
        V = howard_step(W, k_policy, b_policy, par, sol)

    return V


