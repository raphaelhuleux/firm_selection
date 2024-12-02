import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_1d import interp_1d
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from setup import * 
from consav import golden_section_search

""" 
Nested-VFI 
"""
   
@njit 
def bellman_keep(b_next, coh, k, iz, delta, cf, b_grid, k_grid, W):
    div = coh + b_next - cf
    k_next = (1-delta) * k

    if div < 0:
        return np.inf 
    
    V = div + interp_2d(b_grid, k_grid, W[iz], b_next, k_next)
    
    return -V

# definition of coh = z k^alpha - b(1+r) -> don't take into account (1-delta)*k because of irreversibility constraint 
@njit
def solve_keep(W, delta, cf, nu, alpha, r, z_grid, b_grid, k_grid, coh_grid):

    N_z, N_b, N_k = W.shape
    N_coh = coh_grid.size

    V_new = np.zeros((N_z, N_b, N_k))
    k_policy = np.zeros((N_z, N_b, N_k))
    b_policy = np.zeros((N_z, N_b, N_k))

    for iz in range(N_z):
        z = z_grid[iz]
        for icoh in range(N_coh):
            coh = coh_grid[icoh]
            for ik in range(N_k): 
                if exit[iz,icoh,ik]:
                    V_new[iz,icoh,ik] = 0
                    k_policy[iz,icoh,ik] = 0
                    b_policy[iz,icoh,ik] = 0
                else:
                    k = k_grid[ik]
                    k_next = (1-delta) * k

                    b_min = -coh + cf + 1e-6
                    if b_min < b_grid[0]:
                        b_min = b_grid[0]
                    b_max = nu * k
                    b_policy[iz,icoh,ik] = golden_section_search.optimizer(bellman_keep, b_min, b_max, args = (coh, k, iz, delta, cf, b_grid, k_grid, W))
                    k_policy[iz,icoh,ik] = k_next
                    V_new[iz,icoh,ik] = -bellman_keep(b_policy[iz,icoh,ik], coh, k, iz, delta, cf, b_grid, k_grid, W)

    return V_new, k_policy, b_policy

@njit
def bellman_adj(k_next, b, k, iz, alpha, delta, psi, xi, r, z_grid, k_grid, b_grid, V_keep):
    z = z_grid[iz]

    # Compute b_equivalent 
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    coh_adj = z * k**alpha - b * (1+r) + (1-delta) * k - adj_cost

    V = interp_2d(coh_grid, k_grid, V_keep[iz], coh_adj, k_next / (1-delta))
    return -V

@njit
def dividend_constraint(k_next, b_keep, b, k, iz, alpha, cf, delta, psi, xi, r, z_grid, k_grid, b_grid):
    z = z_grid[iz]

    # Compute b_equivalent 
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    y_diff = z * (k_next / (1-delta))**alpha - z * k**alpha
    b_tilde = b + (adj_cost + y_diff + k_next - (1-delta) * k) / (1+r)

    # Get b_next
    b_next = interp_2d(b_grid, k_grid, b_keep[iz], b_tilde, k_next / (1-delta))
    
    # Compute dividends
    coh = z * k**alpha - b * (1+r) + (1-delta) * k
    div = coh + b_next - adj_cost - k_next - cf
    return div 

from quantecon.optimize.root_finding import brentq 

@njit
def solve_adj(V_keep, b_keep, alpha, psi, xi, cf, delta, r, z_grid, b_grid, k_grid):
    N_z, N_b, N_k = V_keep.shape

    V_new = np.zeros((N_z, N_b, N_k))
    k_policy =  np.zeros((N_z, N_b, N_k))
    b_policy =  np.zeros((N_z, N_b, N_k))

    for iz in range(N_z):
        z = z_grid[iz]
        for ib in range(N_b):
            b = b_grid[ib]
            for ik in range(N_k): 
                
                if exit[iz,ib,ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0
                else:
                    k = k_grid[ik]
                    k_min = (1-delta) * k + 1e-8
                    k_max = k_grid[-1]

                    div_min = dividend_constraint(k_min, b_keep, b, k, iz, alpha, cf, delta, psi, xi, r, z_grid, k_grid, b_grid)

                    if div_min < 0:
                        V_new[iz, ib, ik] = 0 
                        k_policy[iz, ib, ik] = 0
                        b_policy[iz, ib, ik] = 0
                    else:
                        res = brentq(dividend_constraint, k_min, 100, args = (b_keep, b, k, iz, alpha, cf, delta, psi, xi, r, z_grid, k_grid, b_grid))
                        k_max = res.root 

                        k_opt = golden_section_search.optimizer(bellman_adj, k_min, k_max, args = ( b, k, iz, alpha, cf, delta, psi, xi, r, z_grid, k_grid, b_grid, b_keep, V_keep))
                        adj_cost = psi / 2 * (k_opt - (1-delta)*k)**2 / k + xi * k 
                        y_diff = z * (k_opt / (1-delta))**alpha - z * k**alpha
                        b_tilde = b + (adj_cost + y_diff + k_opt - (1-delta) * k) / (1+r)

                        k_policy[iz, ib, ik] = k_opt
                        b_policy[iz, ib, ik] = interp_2d(b_grid, k_grid, b_keep[iz], b_tilde, k_opt / (1-delta))
                        coh = z * k**alpha - b * (1+r) + (1-delta) * k
                        div = coh - adj_cost + b_policy[iz, ib, ik] - k_opt - adj_cost - cf
                        
                        if div < 0:
                            print('dividend negative at iz = ', iz, 'ib = ', ib, 'ik = ', ik)
                            print('div = ', div)
                            print(' ')
                    
                        V_new[iz, ib, ik] = -bellman_adj(k_policy[iz, ib, ik],  b, k, iz, alpha, cf, delta, psi, xi, r, z_grid, k_grid, b_grid, b_keep, V_keep)

    return V_new, k_policy, b_policy


def nvfi_step(V, beta, psi, xi, delta, alpha, cf, r, nu, P, z_grid, b_grid, k_grid):
    W = beta * fast_expectation(P, V)
    V_keep, k_keep, b_keep = solve_keep(W, delta, cf, nu, alpha, r, z_grid, b_grid, k_grid)
    V_adj, k_adj, b_adj = solve_adj(V_keep, b_keep, alpha, psi, xi, cf, delta, r, z_grid, b_grid, k_grid)

    k_policy = np.where(V_keep >= V_adj, k_keep, k_adj)
    b_policy = np.where(V_keep >= V_adj, b_keep, b_adj)
    V_new = np.maximum(V_keep, V_adj)

    return V_new, k_policy, b_policy
    
def bellman_howard_nfi(b_next, k_next, b, k, z, iz, psi, xi, delta, alpha, r, cf, b_grid, k_grid, W):
    coh = z * k**alpha - b * (1+r)
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh + b_next - cf - adj_cost - k_next    
    V = div + interp_2d(b_grid, k_grid, W[iz], b_next, k_next)
    
    return V 


def howard_step_nvfi(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):
    N_z, N_b, N_k = W.shape
    V_new = np.zeros((N_z, N_b, N_k))

    for iz in range(N_z):
        z = z_grid[iz]
        for ib in range(N_b):
            b = b_grid[ib]
            for ik in range(N_k): 
                
                if exit[iz,ib,ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0

                else:
                    k = k_grid[ik]
                    k_next = k_policy[iz, ib, ik]
                    b_next = b_policy[iz, ib, ik]
                    V_new[iz,ib,ik] = bellman_howard_nfi(b_next, k_next, b, k, z, iz, psi, xi, delta, alpha, r, cf, b_grid, k_grid, W)
    
    return V_new

def howard_nvfi(V, k_policy, b_policy, beta, P, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    for n in range(30):
        W = beta * fast_expectation(P, V)
        V = howard_step_nvfi(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
    return V 

def nvfi(V_init, beta, nu, psi, xi, delta, alpha, cf, r, P, z_grid, b_grid, k_grid, tol = 1e-5):
    error = 1

    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = nvfi_step(V, beta, psi, xi, delta, alpha, cf, r, nu, P, z_grid, b_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        #V = howard_nvfi(V, k_policy, b_policy, beta, P, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

    return V, k_policy, b_policy 


