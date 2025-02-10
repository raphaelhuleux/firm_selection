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
from setup import fast_expectation, compute_adjustment_cost

""" 
Nested-VFI 
"""
   
@njit 
def bellmankeep(b_next, b, k, iz, r, alpha, delta, cf, z_grid, b_grid, k_grid, W):
    coh = k**alpha * z_grid[iz] - b 
    k_next = (1-delta) * k
    div = coh + b_next - cf

    penalty = 0.0 
    if div < 0.0:
        penalty = -np.abs(div*1e6)
    
    V = div + interp_2d(b_grid, k_grid, W[iz], b_next, k_next)
    
    return V + penalty


@njit
def solve_keep(W, delta, cf, nu, alpha, r, z_grid, b_grid, k_grid):

    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy = np.zeros((Nz, Nb, Nk))
    b_policy = np.zeros((Nz, Nb, Nk))

    for iz in range(Nz):
        z = z_grid[iz]
        for ib in range(Nb):
            b = b_grid[ib]
            for ik in range(Nk): 
                if exit_keep[iz,ib,ik]:
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                else:
                    k = k_grid[ik]
                    k_next = (1-delta) * k
                    coh = z * k**alpha - b * (1+r)

                    b_next =  min(max(-coh + cf , b_grid[0]), nu * k_next)
                    #b_min = max(-coh + cf , b_grid[0])
                    #b_max = nu * (1-delta) * k 

                    #V_b_min = bellmaNkeep(b_min, b, k, iz, r, alpha, delta, cf, z_grid, b_grid, k_grid, W)
                    #V_b_max = bellmaNkeep(b_max, b, k, iz, r, alpha, delta, cf, z_grid, b_grid, k_grid, W)

                    #if V_b_min > V_b_max:
                    #    b_policy[iz,ib,ik] = b_min
                    #else:
                    #    b_policy[iz,ib,ik] = b_max 

                    k_policy[iz,ib,ik] = k_next
                    b_policy[iz,ib,ik] = b_next
                    #V_new[iz,ib,ik] = max(V_b_min, V_b_max)
                    V_new[iz,ib,ik] = bellmankeep(b_next, b, k, iz, r, alpha, delta, cf, z_grid, b_grid, k_grid, W)

    return V_new, k_policy, b_policy


@njit
def bellman_adj(k_next, b, k, iz, alpha, delta, psi, xi, r, cf, nu, z_grid, k_grid, b_grid, W):

    z = z_grid[iz]
    coh = z * k**alpha - b * (1+r) + (1-delta) * k 
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 

    # Compute div_policy
    b_next = b_next_analytical(k_next, b, k, iz, r, nu, delta, alpha, cf, psi, xi, b_grid, k_grid, z_grid, W)
    div = coh + b_next - adj_cost - k_next - cf 

    penalty = 0.0
    if div < 0:
        penalty += np.abs(div*1e6)

    V = div + interp_2d(b_grid, k_grid, W[iz], b_next, k_next)

    return -V + penalty

@njit 
def b_next_analytical(k_next, b, k, iz, r, nu, delta, alpha, cf, psi, xi, b_grid, k_grid, z_grid, W):
    z = z_grid[iz]
    coh = z * k**alpha - b * (1+r) + (1-delta) * k
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi)

    return min(max(-coh + cf + adj_cost + k_next, b_grid[0]), nu * k_next)
    """ 
    b_next_max = nu * k_next

    div_min = coh + b_next_min - adj_cost - k_next - cf
    div_max = coh + b_next_max - adj_cost - k_next - cf

    V_div_min = div_min + interp_2d(b_grid, k_grid, W[iz], b_next_min, k_next)
    V_div_max = div_max + interp_2d(b_grid, k_grid, W[iz], b_next_max, k_next)

    if V_div_min > V_div_max:
        b_next = b_next_min
    else: 
        b_next = b_next_max 

    return b_next
    """

@njit
def dividend_constraint(k_next, b, k, iz, alpha, cf, delta, psi, xi, nu, r, z_grid, k_grid, b_grid, W):
    z = z_grid[iz]

    # Compute b_equivalent 
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi)
    b_next = b_next_analytical(k_next, b, k, iz, r, nu, delta, alpha, cf, psi, xi, b_grid, k_grid, z_grid, W)
    
    # Compute div_policy
    coh = z * k**alpha - b * (1+r) + (1-delta) * k
    div = coh + b_next - adj_cost - k_next - cf
    return div 

@njit
def solve_adj(W, alpha, psi, xi, delta, cf, nu, r, z_grid, b_grid, k_grid):
    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy =  np.zeros((Nz, Nb, Nk))
    b_policy =  np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = z_grid[iz]
        for ib in range(Nb):
            b = b_grid[ib]
            for ik in range(Nk): 
                if exit_adj[iz,ib,ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0
                else:
                    k = k_grid[ik]
                    k_min = (1-delta) * k 
                    k_max = k_max_adj[iz,ib,ik]

                    div_max = dividend_constraint(k_min, b, k, iz, alpha, cf, delta, psi, xi, nu, r, z_grid, k_grid, b_grid, W)
                    div_min = dividend_constraint(k_max, b, k, iz, alpha, cf, delta, psi, xi, nu, r, z_grid, k_grid, b_grid, W)

                    if div_max < 0:
                        V_new[iz, ib, ik] = 0 
                        k_policy[iz, ib, ik] = 0
                        b_policy[iz, ib, ik] = 0
                    else:
                        #k_max = k_max_adj[iz,ib,ik] # k_grid[-1]
                        #if div_min < 0:
                            #res = qe.optimize.root_finding.brentq(dividend_constraint, (1-delta)*k, k_max, args = (b, k, iz, alpha, cf, delta, psi, xi, nu, r, z_grid, k_grid, b_grid, W))  
                            #k_max_eff = res.root
                        #else:
                        k_max_eff = k_max   
                        k_opt = golden_section_search.optimizer(bellman_adj, k_min, k_max_eff, args = (b, k, iz, alpha, delta, psi, xi, r, cf, nu, z_grid, k_grid, b_grid, W))
        
                        k_policy[iz, ib, ik] = k_opt
                        b_policy[iz, ib, ik] = b_next_analytical(k_opt, b, k, iz, r, nu, delta, alpha, cf, psi, xi, b_grid, k_grid, z_grid, W)
                        V_new[iz, ib, ik] = -bellman_adj(k_policy[iz, ib, ik],b, k, iz, alpha, delta, psi, xi, r, cf, nu, z_grid, k_grid, b_grid, W)
                        #V_new[iz,ib,ik] = bellman_invest(b_policy[iz, ib, ik], k_policy[iz, ib, ik], b, k, z, iz, psi, xi, delta, alpha, r, cf, b_grid, k_grid, W)
                        
    return V_new, k_policy, b_policy

def nvfi_step_analytical(V, beta, psi, xi, delta, alpha, cf, r, nu, P, z_grid, b_grid, k_grid):
    W = beta * fast_expectation(P, V)
    V_keep, k_keep, b_keep = solve_keep(W, delta, cf, nu, alpha, r, z_grid, b_grid, k_grid)
    V_adj, k_adj, b_adj = solve_adj(W, alpha, psi, xi, delta, cf, nu, r, z_grid, b_grid, k_grid)

    k_policy = np.where(V_keep >= V_adj, k_keep, k_adj)
    b_policy = np.where(V_keep >= V_adj, b_keep, b_adj)
    V_new = np.maximum(V_keep, V_adj)

    return V_new, k_policy, b_policy
    
@njit
def bellman_invest(b_next, k_next, b, k, z, iz, psi, xi, delta, alpha, r, cf, b_grid, k_grid, W):
    coh = z * k**alpha - b * (1+r) + (1-delta) * k
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh + b_next - k_next - adj_cost - cf
    V = div + interp_2d(b_grid, k_grid, W[iz], b_next, k_next)
    
    return V 

@njit(parallel = True)
def howard_step_nvfi(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):
    Nz, Nb, Nk = W.shape
    V_new = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = z_grid[iz]
        for ib in range(Nb):
            b = b_grid[ib]
            for ik in range(Nk):                 
                if exit_keep[iz,ib,ik]:
                    V_new[iz, ib, ik] = 0

                else:
                    k = k_grid[ik]
                    k_next = k_policy[iz, ib, ik]
                    b_next = b_policy[iz, ib, ik]

                    if k_next == (1-delta) * k:
                        V_new[iz, ib, ik] = bellmaNkeep(b_next, b, k, iz, r, alpha, delta, cf, z_grid, b_grid, k_grid, W)
                    else:
                        V_new[iz, ib, ik] = bellman_invest(b_next, k_next, b, k, z, iz, psi, xi, delta, alpha, r, cf, b_grid, k_grid, W)
    return V_new

def howard_nvfi(V, k_policy, b_policy, beta, P, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid, tol = 1e-4, iter_max = 1000):

    for n in range(100):
        W = beta * fast_expectation(P, V)
        V = howard_step_nvfi(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

    return V 

def solve_nvfi_analytical(V_init, beta, nu, psi, xi, delta, alpha, cf, r, P, z_grid, b_grid, k_grid, tol = 1e-4):
    error = 1

    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = nvfi_step_analytical(V, beta, psi, xi, delta, alpha, cf, r, nu, P, z_grid, b_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        V = howard_nvfi(V, k_policy, b_policy, beta, P, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

    inaction = (k_policy == (1-delta)*k_grid[np.newaxis,np.newaxis,:])
    coh = z_grid[:,np.newaxis,np.newaxis] * k_grid[np.newaxis,np.newaxis,:]**alpha + (1-delta) * k_grid[np.newaxis,np.newaxis,:] - b_grid[np.newaxis,:,np.newaxis] * (1+r) 
    adj_cost = compute_adjustment_cost(k_policy, k_grid[np.newaxis,np.newaxis,:], psi, xi) 
    div_opt = (1-exit_keep) * (coh - adj_cost*(1-inaction) - k_policy + b_policy - cf )

    return V, k_policy, b_policy, inaction, div_opt