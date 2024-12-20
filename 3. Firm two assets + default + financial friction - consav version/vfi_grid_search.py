import numpy as np 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
from model_functions import * 

"""
VFI 
"""

@njit 
def bellman_invest(b_next, k_next, b, k, iz, par, sol, W):
    z = par.z_grid[iz]

    coh = z * k**par.alpha + (1-par.delta) * k - b 
    q = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next * q - par.cf

    if b_next > par.nu * k_next:
        return -np.inf
    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V

@njit 
def bellman_inaction(b_next, b, k, iz, par, sol, W):
    z = par.z_grid[iz]
    k_next = (1-par.delta) * k

    coh = z * k**par.alpha - b
    q = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
    div = coh + q * b_next - par.cf

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V


@njit
def grid_search_invest(b, k, iz, k_max, W, par, sol, Nb_choice = 60, Nk_choice = 100):

    Vmax = -np.inf 

    k_min = (1-par.delta) * k + 1e-8
    k_choice = np.linspace(k_min, k_max, Nk_choice)

    for k_next in k_choice:
        b_max = par.nu * k_next 
        b_choice = np.linspace(par.b_grid[0], b_max, Nb_choice)
        for b_next in b_choice:
            V = bellman_invest(b_next, k_next, b, k, iz, par, sol, W)
            q = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
            if q == 0.0:
                break
            if V > Vmax:
                Vmax = V
                b_opt = b_next
                k_opt = k_next

    return Vmax, b_opt, k_opt

@njit
def grid_search_inaction(b, k, iz, b_min, W, par, sol, Nb_choice = 60):

    Vmax = -np.inf 
    b_max = par.nu * (1-par.delta) * k
    b_choice = np.linspace(b_min, b_max, Nb_choice)
    b_opt = 0.0
    
    for b_next in b_choice:
        V = bellman_inaction(b_next, b, k, iz, par, sol, W)
        q = debt_price_function(iz, (1-par.delta)*k, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
        if q == 0.0:
            break

        if V > Vmax:
            Vmax = V
            b_opt = b_next
    
    return Vmax, b_opt


@njit(parallel = True)
def vfi_step(V, par, sol):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    b_policy = np.empty_like(V)

    Nz, Nb, Nk = V.shape

    W = par.beta * fast_expectation(par.P, V)

    for iz in prange(Nz):
        for ik in range(Nk):
            for ib in range(Nb):

                if sol.exit_policy[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0.0
                    k_policy[iz, ib, ik] = 0.0
                    b_policy[iz, ib, ik] = 0.0
                else:
                    b = par.b_grid[ib]
                    k = par.k_grid[ik]

                    k_max = sol.k_max_adj[iz, ib, ik]
                    Vinv, b_inv, k_inv = grid_search_invest(b, k, iz, k_max, W, par, sol, Nb_choice = par.Nb_choice, Nk_choice = par.Nk_choice)

                    b_min_keep = sol.b_min_keep[iz, ib, ik]
                    Vina, b_ina = grid_search_inaction(b, k, iz, b_min_keep, W, par, sol, Nb_choice = par.Nb_choice)

                    if Vinv > Vina:
                        V_new[iz, ib, ik] = Vinv
                        k_policy[iz, ib, ik] = k_inv
                        b_policy[iz, ib, ik] = b_inv
                    else:   
                        V_new[iz, ib, ik] = Vina
                        k_policy[iz, ib, ik] = (1-par.delta) * k
                        b_policy[iz, ib, ik] = b_ina

              
    return V_new, k_policy, b_policy

def solve_vfi_grid_search(par, sol, tol = 1e-4, do_howard = True):
    error = 1

    V_init = np.zeros((par.Nz, par.Nb, par.Nk))
    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = vfi_step(V, par, sol)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard(V, k_policy, b_policy, par, sol)

    sol.inaction[...] = k_policy > (1-par.delta) * par.k_grid[None, None, :]
    sol.k_policy[...] = k_policy
    sol.b_policy[...] = b_policy
    sol.V[...] = V
    compute_optimal_div_policy(b_policy, k_policy, par, sol)


@njit(parallel = True)
def howard_step(W, k_policy, b_policy, par, sol):

    V_new = np.empty_like(W)
    Nz, Nb, Nk = W.shape

    for iz in prange(Nz):
        for ik in range(Nk):
            for ib in range(Nb):
                if sol.exit_policy[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0.0
                else:
                    b = par.b_grid[ib]
                    k = par.k_grid[ik]
                    b_next = b_policy[iz, ib, ik]
                    k_next = k_policy[iz, ib, ik]
                    
                    if k_next == (1-par.delta) * k:
                        V_new[iz, ib, ik] = bellman_inaction(b_next, b, k, iz, par, sol, W)
                    else:
                        V_new[iz, ib, ik] = bellman_invest(b_next, k_next, b, k, iz, par, sol, W)
                    
    return V_new

def howard(V, k_policy, b_policy, par, sol):

    for n in range(50):
        W = par.beta * fast_expectation(par.P, V)
        V = howard_step(W, k_policy, b_policy, par, sol)

    return V


