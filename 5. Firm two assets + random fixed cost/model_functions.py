
import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d, interp_2d_vec
from numba import njit, prange
import quantecon as qe 
from consav.golden_section_search import optimizer 


@njit 
def compute_adjustment_cost(k_next, k, delta, psi, xi):
    return psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k

@njit
def dividend_keep(b_next, k_next, exit_policy, z, b, k, iz, par):
    q = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par.P, par.k_grid, par.b_grid)
    div = z * k**par.alpha + q * b_next - b - par.cf  
    return div

@njit
def dividend_adj(b_next, k_next, exit_policy, z, b, k, iz, par):
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
    q = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par.P, par.k_grid, par.b_grid)
    div = z * k**par.alpha + q * b_next - b + (1-par.delta) * k - k_next - adj_cost - par.cf  
    return div

@njit 
def compute_optimal_div_policy(b_policy, k_policy, par, sol):
    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    z_grid, b_grid, k_grid = par.z_grid, par.b_grid, par.k_grid

    div_policy_max = sol.div_policy 

    for iz in range(Nz):
        z = z_grid[iz]
        for ik in range(Nk):
            k = k_grid[ik]
            for ib in range(Nb):
                b = b_grid[ib] 

                if sol.exit_policy[iz,ib,ik] == 1:
                    div_policy_max[iz,ib,ik] = -np.inf

                else:
                    b_next = b_policy[iz,ib,ik]
                    k_next = k_policy[iz,ib,ik]
                    if k_next == (1-par.delta) * k:
                        div_policy_max[iz,ib,ik] = dividend_keep(b_next, k_next, sol.exit_policy, z, b, k, iz, par)
                    else: 
                        div_policy_max[iz,ib,ik] = dividend_adj(b_next, k_next, sol.exit_policy, z, b, k, iz, par)


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
def debt_price_function(iz, k_next, b_next, r, exit_policy, par):
    q = 0.0
    k_next = np.ones_like(par.omega_grid) * k_next
    
    for iz_prime in range(par.Nz):
        b_next_tilde = b_next + par.omega_grid
        Pz = par.P[iz,iz_prime] * par.omega_p
        exit_prob = np.zeros_like(b_next_tilde)
        interp_2d_vec(par.b_grid, par.k_grid, exit_policy[iz_prime,:,:], b_next_tilde, k_next, exit_prob)
        q_temp =  np.sum(1/(1+r) * (1 - exit_prob) * Pz) 
        q += q_temp 

        """
    for iz_prime in range(par.Nz):
        for i_omega in range(par.Nomega):
            omega = par.omega_grid[i_omega]
            b_next_tilde = b_next + omega 
            Pz = par.P[iz,iz_prime] * par.omega_p[i_omega]	
            exit_prob = interp_2d(par.b_grid, par.k_grid, exit_policy[iz_prime,:,:], b_next_tilde, k_next)
            q_temp =  1/(1+r) * (1 - exit_prob)  # assuming the bank cannot recover any assets in case of default
            q += Pz * q_temp
        """
    
    return q 