
import numpy as np 
import matplotlib.pyplot as plt
from consav.linear_interp_1d import interp_1d, interp_1d_vec
from consav.linear_interp_2d import interp_2d, interp_2d_vec
from numba import njit, prange
import quantecon as qe 


@njit 
def compute_adjustment_cost(k_next, k, delta, psi, xi):
    return psi / 2 * (k_next - (1-delta)*k)**2 / (k+1e-6) + xi * k


@njit
def dividend_fun(b_next, k_next, z, b, k, iz, sol, par):
    if k_next == (1-par.delta) * k:
        adj_cost = 0
    else:
        adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
    q = interp_2d(par.b_grid, par.k_grid, sol.q[iz], b_next, k_next)
    div = z * k**par.alpha + q * b_next - b + (1-par.delta) * k - k_next - adj_cost
    return div

@njit 
def compute_optimal_div_policy(b_policy, k_policy, par, sol):
    Nz, Nb, Nk = par.Nz, par.Nb, par.Nk
    z_grid, b_grid, k_grid = par.z_grid, par.b_grid, par.k_grid

    for iz in range(Nz):
        z = z_grid[iz]
        for ik in range(Nk):
            k = k_grid[ik]
            for ib in range(Nb):
                b = b_grid[ib] 

                if sol.exit_policy[iz,ib,ik] == 1:
                    sol.div_policy[iz,ib,ik] = -np.inf

                else:
                    b_next = b_policy[iz,ib,ik]
                    k_next = k_policy[iz,ib,ik]
                    sol.div_policy[iz,ib,ik] = dividend_fun(b_next, k_next, z, b, k, iz, sol, par)


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

@njit(parallel = True)
def compute_expectation_omega(V, par):

    W = np.zeros((par.Nz, par.Nb, par.Nk))

    for iz in prange(par.Nz):
        for ik in range(par.Nk):
            for ib in range(par.Nb):
                temp = np.zeros_like(par.omega_grid)
                interp_1d_vec(par.b_grid, V[iz,:,ik], par.b_grid[ib] + par.omega_grid, temp)
                W[iz,ib,ik] = np.sum(par.omega_p * temp)

    return W

@njit
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
    
    return q 

def multiply_ith_dimension(Pi, i, X):
    """If Pi is a matrix, multiply Pi times the ith dimension of X and return"""
    
    X = np.swapaxes(X, 0, i)
    shape = X.shape
    X = X.reshape(shape[0], -1)

    # iterate forward using Pi
    X = Pi @ X

    # reverse steps
    X = X.reshape(Pi.shape[0], *shape[1:])
    return np.swapaxes(X, 0, i)