import numpy as np 
from consav.linear_interp_2d import interp_2d
from consav.linear_interp_3d import interp_3d

import numba as nb
from model_functions import * 
from precompute import objective_dividend_keeper
import quantecon as qe 
from consav.golden_section_search import optimizer 

""" 
Problem with the code: 
when solving the adjuster problem, some intermediate values of k_next between k_min and k_max are not feasible,
even if k_min and k_max are feasible! this means that there is some strong non-monotonicity in the problem: 
increasing k_next might actually make you more.
Maybe the issue is with the price function q that looks like a step function
Also try to redo compute_exit_decision_step with a dense grid_search?
"""

"""
VFI 
"""

""" 
Solve keeper problem 
"""

@njit 
def bellman_keep(b_next, b, k, iz, W, par, sol):
    z = par.z_grid[iz]
    coh = k**par.alpha * z - b 
    k_next = (1-par.delta) * k
    q = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.q, z, b_next, k_next)
    div = coh + b_next * q - par.cf

    penalty = 0.0 
    if div < 0.0:
        penalty = -np.abs(div*1e6)
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V + penalty

@njit(parallel = True)
def solve_keep(W, par, sol):

    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy = np.zeros((Nz, Nb, Nk))
    b_policy = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk): 
                if sol.exit_policy[iz,ib,ik]:
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                else:
                    k = par.k_grid[ik]
                    k_next = (1-par.delta) * k
                    b_next =  sol.b_min_keep[iz,ib,ik]

                    k_policy[iz,ib,ik] = k_next
                    b_policy[iz,ib,ik] = b_next
                    V_new[iz,ib,ik] = bellman_keep(b_next, b, k, iz, W, par, sol)

    return V_new, k_policy, b_policy

""" 
Solve adjuster problem
"""

@nb.njit
def objective_dividend_adj(b_next, k_next, adj_cost, z, b, k, sol, par):
    #q = debt_price_function(iz, k_next, b_next, par.r, exit_policy, par.P, par.k_grid, par.b_grid)
    q = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.q, z, b_next, k_next)
    div = z * k**par.alpha + q * b_next - b - par.cf + (1-par.delta) * k - adj_cost - k_next
    return -div


@njit 
def b_next_adj(k_next, adj_cost, b, k, iz, par, sol): 

    exit_policy = sol.exit_policy
    z = par.z_grid[iz]
    b_grid = par.b_grid

    div_b_min = -objective_dividend_adj(b_grid[0], k_next, adj_cost, z, b, k, sol, par)
    b_max = par.b_grid[-1]

    if div_b_min < 0:
        div_b_max = -objective_dividend_adj(b_max, k_next, adj_cost, z, b, k, sol, par)
        if div_b_max < 0:
            b_max = optimizer(objective_dividend_adj, b_grid[0], b_max, args=(k_next, adj_cost, z, b, k, sol, par))
            div_b_max = -objective_dividend_adj(b_max, k_next, adj_cost, z, b, k, sol, par)

        res = qe.optimize.root_finding.bisect(objective_dividend_adj, b_grid[0], b_max, args=(k_next, adj_cost, z, b, k, sol, par))
        b_next = res.root
    else:
        b_next = b_grid[0]

    return b_next

"""
iz = 0
ib = 33 
ik = 10 
b = par.b_grid[ib]
k = par.k_grid[ik]

k_min = (1-par.delta) * k + 1e-6
k_max = sol.k_max_adj[iz,ib,ik]

N_k_choice = 1_001
N_b_choice = 1_000
k_choice = np.linspace(k_min, k_max, N_k_choice)
b_choice = np.linspace(0, par.b_grid[-1], N_b_choice)

z = par.z_grid[iz]

coh = z * k**par.alpha + (1-par.delta) * k - b 
adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 

q_vec = np.zeros((N_k_choice,  N_b_choice))
div_vec = np.zeros((N_k_choice, N_b_choice)) 

for i, k_next in enumerate(k_choice):
    for j, b_next in enumerate(b_choice):
        q_vec[i,j] = debt_price_function(iz, k_next, b_next, par.r, sol.exit_policy, par.P, par.k_grid, par.b_grid)
        div_vec[i,j] = z * k**par.alpha + q_vec[i,j] * b_next - b - par.cf + (1-par.delta) * k - adj_cost - k_next


plt.plot(k_choice, np.max(div_vec, axis = 1))
plt.xlabel('k_next')
plt.ylabel('div')
plt.show()

plt.plot(b_choice,q_vec[[0,100,200,300,400,500,600,700,800,900,999],:].T)
plt.xlabel('b_next')
plt.ylabel('div')
plt.show()


plt.plot(b_choice,div_vec[[0,100,200,300,400,500,600,700,800,900,999],:].T)
plt.xlabel('b_next')
plt.ylabel('div')
plt.show()
"""

@njit 
def bellman_adj(k_next, b, k, iz, par, sol, W):
    z = par.z_grid[iz]

    coh = z * k**par.alpha + (1-par.delta) * k - b 
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 

    b_next = b_next_adj(k_next, adj_cost, b, k, iz, par, sol)
    q = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.q, z, b_next, k_next)
    div = coh - adj_cost - k_next + b_next * q - par.cf

    penalty = 0.0
    if div < 0:
        penalty += np.abs(div*1e6)
    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return -V + penalty

@njit 
def bellman_invest(k_next, b_next, b, k, iz, par, sol, W): 

    z = par.z_grid[iz]

    coh = z * k**par.alpha + (1-par.delta) * k - b 
    q = interp_3d(par.z_grid_dense, par.b_grid_dense, par.k_grid_dense, sol.q, z, b_next, k_next)
    adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next * q - par.cf    
    V = div + interp_2d(par.b_grid, par.k_grid, W[iz], b_next, k_next)
    
    return V 


@njit(parallel = True)
def solve_adj(W, par, sol):
    Nz, Nb, Nk = W.shape

    V_new = np.zeros((Nz, Nb, Nk))
    k_policy = np.zeros((Nz, Nb, Nk))
    b_policy = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk):
                k = par.k_grid[ik]

                if sol.exit_policy[iz,ib,ik]:
                    V_new[iz,ib,ik] = 0
                    k_policy[iz,ib,ik] = 0
                    b_policy[iz,ib,ik] = 0
                else:
                    k_min = (1-par.delta) * k + 1e-6
                    k_max = sol.k_max_adj[iz,ib,ik]
                    k_opt = optimizer(bellman_adj, k_min, k_max, args = (b, k, iz, par, sol, W))

                    adj_cost = compute_adjustment_cost(k_opt, k, par.delta, par.psi, par.xi) # psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 

                    b_policy[iz,ib,ik] = b_next_adj(k_opt, adj_cost, b, k, iz, par, sol)
                    k_policy[iz,ib,ik] = k_opt
                    V_new[iz,ib,ik] = bellman_invest(k_opt, b_policy[iz,ib,ik], b, k, iz, par, sol, W)

    return V_new, k_policy, b_policy

""" 
NVFI
"""

def nvfi_step_analytical(V, par, sol):
    W = par.beta * fast_expectation(par.P, V)
    V_keep, k_keep, b_keep = solve_keep(W, par, sol)
    V_adj, k_adj, b_adj = solve_adj(W, par, sol)

    k_policy = np.where(V_keep >= V_adj, k_keep, k_adj)
    b_policy = np.where(V_keep >= V_adj, b_keep, b_adj)
    V_new = np.maximum(V_keep, V_adj)

    return V_new, k_policy, b_policy

def solve_nvfi_analytical(par, sol, tol = 1e-4, do_howard = True):
    error = 1

    V_init = np.zeros((par.Nz, par.Nb, par.Nk))
    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = nvfi_step_analytical(V, par, sol)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard_nvfi(V, k_policy, b_policy, par, sol)


    sol.inaction[...] = k_policy > (1-par.delta) * par.k_grid[None, None, :]
    sol.k_policy[...] = k_policy
    sol.b_policy[...] = b_policy
    sol.V[...] = V
    compute_optimal_div_policy(b_policy, k_policy, par, sol)

""" 
Howard
"""

@njit(parallel = True)
def howard_step_nvfi(W, k_policy, b_policy, par, sol):
    Nz, Nb, Nk = W.shape
    V_new = np.zeros((Nz, Nb, Nk))

    for iz in prange(Nz):
        z = par.z_grid[iz]
        for ib in range(Nb):
            b = par.b_grid[ib]
            for ik in range(Nk):                 
                if sol.exit_policy[iz,ib,ik]:
                    V_new[iz, ib, ik] = 0

                else:
                    k = par.k_grid[ik]
                    k_next = k_policy[iz, ib, ik]
                    b_next = b_policy[iz, ib, ik]

                    if k_next == (1-par.delta) * k:
                        V_new[iz, ib, ik] = bellman_keep(b_next, b, k, iz, W, par, sol)
                    else:
                        V_new[iz, ib, ik] = bellman_invest(k_next, b_next, b, k, iz, par, sol, W)
    return V_new

def howard_nvfi(V, k_policy, b_policy, par, sol, tol = 1e-4, iter_max = 1000):

    for n in range(100):
        W = par.beta * fast_expectation(par.P, V)
        V = howard_step_nvfi(W, k_policy, b_policy, par, sol)

    return V 