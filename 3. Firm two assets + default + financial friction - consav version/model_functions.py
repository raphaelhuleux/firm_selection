
import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from consav.golden_section_search import optimizer 


@njit 
def compute_adjustment_cost(k_next, k, delta, psi, xi):
    return psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k

@njit
def dividend_keep(b_next, k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid):
    q = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
    div = z * k**alpha + q * b_next - b - cf  
    return div

@njit
def dividend_adj(b_next, k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid):
    adj_cost = compute_adjustment_cost(k_next, k, delta, psi, xi)
    q = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
    div = z * k**alpha + q * b_next - b - + (1-delta) * k - k_next - adj_cost - cf  
    return div

@njit 
def compute_optimal_dividends(b_policy, k_policy, par, sol):
    N_z, N_b, N_k = par.N_z, par.N_b, par.N_k
    z_grid, b_grid, k_grid = par.z_grid, par.b_grid, par.k_grid

    dividends_max = sol.dividends 

    for iz in range(N_z):
        z = z_grid[iz]
        for ik in range(N_k):
            k = k_grid[ik]
            for ib in range(N_b):
                b = b_grid[ib] 

                if exit_policy[iz,ib,ik] == 1:
                    dividends_max[iz,ib,ik] = -np.inf

                else:
                    b_next = b_policy[iz,ib,ik]
                    k_next = k_policy[iz,ib,ik]
                    if k_next == (1-delta) * k:
                        dividends_max[iz,ib,ik] = dividend_keep(b_next, k_next, sol.exit_policy, z, b, k, iz, par.r, par.cf, par.P, par.k_grid, par.b_grid)
                    else: 
                        dividends_max[iz,ib,ik] = dividend_adj(b_next, k_next, sol.exit_policy, z, b, k, iz, par.r, par.cf, par.P, par.k_grid, par.b_grid)
@njit
def fast_expectation(Pi, X):
    
    res = np.zeros_like(X)
    X = np.ascontiguousarray(X)
    
    for i in range(Pi.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                for l in range(X.shape[0]):
                    res[i,j,k] += Pi[i,l] * X[l,j,k]
                            
    return res

@nb.njit
def debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid):
    q = 0.0
    N_z = P.shape[0]

    for iz_prime in range(N_z):
        Pz = P[iz,iz_prime] 
        exit_prob = interp_2d(b_grid, k_grid, exit[iz_prime,:,:], b_next, k_next)
        q_temp =  1/(1+r) * (1 - exit_prob)  # assuming the bank cannot recover any assets in case of default
        q += Pz * q_temp
    
    return q 