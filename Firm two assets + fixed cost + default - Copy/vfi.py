import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from setup import * 
from setup import fast_expectation, compute_adjustment_cost

"""
VFI 
"""

@njit 
def bellman_invest(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W):
    z = z_grid[iz]

    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf

    if b_next > nu * k_next:
        return -np.inf
    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(b_grid,k_grid, W[iz], b_next, k_next)
    
    return V

@njit 
def bellman_inaction(b_next, b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W):
    z = z_grid[iz]
    k_next = (1-delta) * k

    coh = z * k**alpha - b * (1+r)
    div = coh + b_next - cf

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(b_grid,k_grid, W[iz], b_next, k_next)
    
    return V


@njit
def grid_search_invest(b, k, iz, alpha, delta, psi, xi, nu, r, cf, z_grid, b_grid, k_grid, W, Nb_choice = 100, Nk_choice = 100):

    Vmax = -np.inf 

    k_min = (1-delta) * k + 1e-8
    k_max = k_grid[-1]
    k_choice = np.linspace(k_min, k_max, Nk_choice)

    b_min = b * (1+r) - z_grid[iz] * k**alpha + cf
    if b_min < b_grid[0]:
        b_min = b_grid[0]
    b_max = nu * k_max 
    b_choice = np.linspace(b_min, b_max, Nb_choice)

    for ik_next in range(Nk_choice):
        k_next = k_choice[ik_next]
        for ib_next in range(Nb_choice):
            b_next = b_choice[ib_next]
            V = bellman_invest(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
            if V > Vmax:
                Vmax = V
                b_max = b_next
                k_max = k_next

    return Vmax, b_max, k_max

@njit
def grid_search_inaction(b, k, iz, alpha, delta, nu, r, cf, z_grid, b_grid, k_grid, W, Nb_choice = 100):

    Vmax = -np.inf 
    b_min = b * (1+r) - z_grid[iz] * k**alpha 
    if b_min < b_grid[0]:
        b_min = b_grid[0]
    b_max = nu * (1-delta) * k
    b_choice = np.linspace(b_min, b_max, Nb_choice)
    
    for ib_next in range(Nb_choice):
        b_next = b_choice[ib_next]
        V = bellman_inaction(b_next, b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W)
        if V > Vmax:
            Vmax = V
            b_max = b_next
    
    return Vmax, b_max


@njit(parallel = True)
def vfi_step(V, beta, psi, xi, delta, alpha, cf, r, nu, P, z_grid, b_grid, k_grid):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    b_policy = np.empty_like(V)

    N_z, N_b, N_k = V.shape

    W = beta * fast_expectation(P, V)

    for iz in prange(N_z):
        for ik in range(N_k):
            for ib in range(N_b):

                if exit_keep[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0
                else:
                    b = b_grid[ib]
                    k = k_grid[ik]

                    Vinv, b_inv, k_inv = grid_search_invest(b, k, iz, alpha, delta, psi, xi, nu, r, cf, z_grid, b_grid, k_grid, W)
                    Vina, b_ina = grid_search_inaction(b, k, iz, alpha, delta, nu, r, cf, z_grid, b_grid, k_grid, W)

                    if Vinv > Vina:
                        V_new[iz, ib, ik] = Vinv
                        k_policy[iz, ib, ik] = k_inv
                        b_policy[iz, ib, ik] = b_inv
                    else:   
                        V_new[iz, ib, ik] = Vina
                        k_policy[iz, ib, ik] = (1-delta) * k
                        b_policy[iz, ib, ik] = b_ina

              
    return V_new, k_policy, b_policy


@njit(parallel = True)
def howard_step(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    V_new = np.empty_like(W)
    N_z, N_b, N_k = W.shape

    for iz in prange(N_z):
        for ik in range(N_k):
            for ib in range(N_b):
                if exit_keep[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                else:
                    b = b_grid[ib]
                    k = k_grid[ik]
                    b_next = b_policy[iz, ib, ik]
                    k_next = k_policy[iz, ib, ik]
                    
                    if k_next == (1-delta) * k:
                        V_new[iz, ib, ik] = bellman_inaction(b_next, b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W)
                    else:
                        V_new[iz, ib, ik] = bellman_invest(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
                    
    return V_new

def howard(V, k_policy, b_policy, beta, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    for n in range(50):
        W = beta * fast_expectation(P, V)
        V = howard_step(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

    return V

def vfi(V_init, beta, nu, psi, xi, delta, alpha, cf, r, P, z_grid, b_grid, k_grid, tol = 1e-4, do_howard = True):
    error = 1

    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = vfi_step(V, beta, psi, xi, delta, alpha, cf, r, nu, P, z_grid, b_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard(V, k_policy, b_policy, beta, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

    inaction = (k_policy == (1-delta)*k_grid[np.newaxis,np.newaxis,:])
    coh = z_grid[:,np.newaxis,np.newaxis] * k_grid[np.newaxis,np.newaxis,:]**alpha + (1-delta) * k_grid[np.newaxis,np.newaxis,:] - b_grid[np.newaxis,:,np.newaxis] * (1+r) 
    adj_cost = compute_adjustment_cost(k_policy, k_grid[np.newaxis,np.newaxis,:], psi, xi) 
    div_opt = (1-exit_keep) * (coh - adj_cost*(1-inaction) - k_policy + b_policy - cf )
    return V, k_policy, b_policy, inaction, div_opt
