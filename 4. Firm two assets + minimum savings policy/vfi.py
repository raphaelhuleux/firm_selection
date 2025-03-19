import numpy as np 
from consav.linear_interp_2d import interp_2d
from consav.linear_interp_3d import interp_3d

from numba import njit, prange
from model_functions import * 
from precompute import objective_dividend_keeper
from quantecon.optimize import nelder_mead
from consav.golden_section_search import optimizer 


"""
VFI 
"""

@njit
def bellman_keep(leverage, z, b, k, q_mat, W, par):
    b_next = leverage * (1-par.delta) * k
    k_next = (1-par.delta) * k
    coh = k**par.alpha * z - b
    q = interp_2d(par.b_grid, par.k_grid, q_mat, b_next, k_next)
    div = coh + b_next * q
    penalty = 0.0
    if div < 0.0:
        penalty += np.abs(div)*1e3
    V = div + interp_2d(par.b_grid, par.k_grid, W, b_next, k_next) 
    return -V + penalty    

@njit
def bellman_adj(x, z, b, k, q_mat, W, par):
    
    k_next, leverage = x 
    coh = k**par.alpha * z - b
    b_next = k_next * leverage
    q = interp_2d(par.b_grid, par.k_grid, q_mat, b_next, k_next)
    adj_cost = compute_adjustment_cost(k, k_next, par.delta, par.psi, par.xi)
    div = coh + b_next * q - k_next - adj_cost + (1-par.delta) * k
    penalty = 0.0
    if div < 0.0:
        penalty += np.abs(div)*1e3
     
    V = div + interp_2d(par.b_grid, par.k_grid, W, b_next, k_next)
    return V - penalty 

@njit
def vfi_step(W, sol, par):
    
    V_new = np.zeros((par.Nz, par.Nb, par.Nk)) 
    k_policy = np.zeros((par.Nz, par.Nb, par.Nk))
    b_policy = np.zeros((par.Nz, par.Nb, par.Nk))

    for iz in range(par.Nz):
        z = par.z_grid[iz]
        for ik in range(par.Nk):
            k = par.k_grid[ik]
            for ib in range(par.Nb):
                b = par.b_grid[ib]

                if sol.exit_policy[iz,ib,ik]:
                    # Exit case
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                elif sol.unconstrained_indicator[iz,ib,ik]:
                    # Unconstrained case
                    k_next = sol.k_policy_unconstrained[iz,ik]
                    b_next = sol.b_policy_unconstrained[iz,ik]
                    k_policy[iz,ib,ik] = k_next
                    b_policy[iz,ib,ik] = b_next
                    div = sol.div_unconstrained[iz,ib,ik] 
                    V_new[iz,ib,ik] = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
                else: 
                    # Do VFI, solve keeper and ajuster 
                    # Keeper 
                    leverage = optimizer(bellman_keep, 0.0, par.nu, args = (z, b, k, sol.q[iz], W[iz], par), tol = 1e-6)
                    V_keep = -bellman_keep(leverage, z, b, k, sol.q[iz], W[iz], par)

                    # Adjuster
                    if sol.exit_policy_adj[iz,ib,ik]:
                        V_adj = -np.inf 
                    else:
                        x0 = np.array([k, 0.5])
                        bounds = np.array([[(1-par.delta)*k, sol.k_max_adj[iz,ib,ik]], [0.0, par.nu]])
                        res = nelder_mead(bellman_adj, x0, bounds = bounds, args = (z, b, k, sol.q[iz], W[iz], par))
                        V_adj = res.fun 

                    if res.success == False:
                        print('Optimization failed')
                        print('iz = ', iz)
                        print('ib = ', ib)
                        print('ik = ', ik)

                    if V_keep > V_adj:
                        V_new[iz,ib,ik] = V_keep
                        k_policy[iz,ib,ik] = k * (1-par.delta)
                        b_policy[iz,ib,ik] = leverage * (1-par.delta) * k
                    else:
                        V_new[iz,ib,ik] = V_adj
                        k_policy[iz,ib,ik] = res.x[0]
                        b_policy[iz,ib,ik] = res.x[0] * res.x[1]

    return V_new, k_policy, b_policy



def solve_vfi_ss(par, ss):

    V = np.zeros((par.Nz, par.Nb, par.Nk))
    tol = 1e-4
    error = 1.0
    it = 0

    print('Solving model with VFI')
    print('----------------------')

    while error > tol:
        it += 1

        W = par.beta * multiply_ith_dimension(par.P, 0, V)
        W = compute_expectation_omega(W, par)
        W = compute_expectation_k_shock(W, par)

        V_new, k_policy, b_policy = vfi_step(W, ss, par)
        error = np.max(np.abs(V_new - V))
        print(error)
        V = V_new 

    print('Done sweetie')
    print(' ')
    return W, k_policy, b_policy

