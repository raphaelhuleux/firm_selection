
import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_1d import interp_1d
from consav.linear_interp_2d import interp_2d, interp_2d_vec
from numba import njit, prange
import quantecon as qe 
from consav.golden_section_search import optimizer 


@nb.njit 
def compute_adjustment_cost(k_next, k, delta, psi, xi):
    return psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k


@nb.njit
def dividend_fun(b_next, k_next, z, b, k, iz, sol, par):
    if k_next == (1-par.delta) * k:
        adj_cost = 0
    else:
        adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = z * k**par.alpha + q * b_next - b + (1-par.delta) * k - k_next - adj_cost
    return div

@nb.njit 
def compute_optimal_div_policy(b_policy, k_policy, par, sol):
    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    z_grid, b_grid, k_grid = par.z_grid, par.b_grid, par.k_grid

    div_policy = sol.div_policy 

    for iz in range(Nz):
        z = z_grid[iz]
        for ik in range(Nk):
            k = k_grid[ik]
            for ib in range(Nb):
                b = b_grid[ib] 

                if sol.exit_policy[iz,ib,ik] == 1:
                    div_policy[iz,ib,ik] = -np.inf

                else:
                    b_next = b_policy[iz,ib,ik]
                    k_next = k_policy[iz,ib,ik]
                    div_policy[iz,ib,ik] = dividend_fun(b_next, k_next, z, b, k, iz, sol, par)


@nb.njit
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
def compute_expectation_omega(V, par):

    W = np.zeros((par.Nz, par.Nb, par.Nk))

    V_temp = 0.0

    for iz in range(par.Nz):
        for ik in range(par.Nk):
            for ib in range(par.Nb):
                V_temp = 0.0
                for i_omega in range(par.Nomega):
                    b_tilde = par.b_grid[ib] + par.omega_grid[i_omega]
                    V_temp += par.omega_p[i_omega] * interp_1d(par.b_grid, V[iz,:,ik], b_tilde)
                W[iz,ib,ik] = V_temp

    return W

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
