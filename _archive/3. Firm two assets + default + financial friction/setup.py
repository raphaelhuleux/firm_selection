import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from consav.golden_section_search import optimizer 

""" 
TODO      
- Do and plot simulations over time
    - Who exits?
- Stationary distribution in partial equilibrium (add intrants)
- Simulate a monetary policy shock (a change in r)
- Add labor? 
"""

# Parameters

alpha = 1/2 # capital share
beta = 0.95 # discount factor
delta = 0.1

rho = 0.8
sigma_z = 0.2
psi = 0.05
xi = 0.001
cf = 0.1

nu = 0.9
r = (1/beta - 1) * 1.1

# Steady state
z_bar = 1
kbar = (alpha * z_bar /(1/beta-1+delta))**(1/(1-alpha))

# Grid
N_k = 80
N_b = 100
N_z = 5

k_min = 0.0
k_max = 2*kbar

b_min = 0
b_max = nu*k_max

b_grid = np.linspace(b_min,b_max,N_b)
k_grid = np.linspace(k_min,k_max,N_k)

shock = qe.rouwenhorst(N_z, rho, sigma_z)
P = shock.P
z_grid = z_bar * np.exp(shock.state_values)


@njit 
def compute_adjustment_cost(k_next, k, psi, xi):
    return psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k

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
def objective_div_max_inv(k_next, z, k, b):
    b_next = nu * k_next
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi) 
    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    div = coh - adj_cost - k_next + b_next - cf
    return div 


@nb.njit(
    nb.float64(
        nb.int64,                # iz
        nb.float64,              # k_next
        nb.float64,              # b_next
        nb.float64,              # r
        nb.float64[:, :, :],     # exit
        nb.float64[:, :],        # P
        nb.float64[:],           # b_grid
        nb.float64[:]            # k_grid
    )
)
def debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid):
    q = 0.0
    N_z = P.shape[0]

    for iz_prime in range(N_z):
        Pz = P[iz,iz_prime] 
        exit_prob = interp_2d(b_grid, k_grid, exit[iz_prime,:,:], b_next, k_next)
        #if exit_prob > 0.5:
        #    exit_prob = 1
        #else:
        #    exit_prob = 0.0 
        q_temp =  1/(1+r) * (1 - exit_prob)  # assuming the bank cannot recover any assets in case of default
        q += Pz * q_temp
    
    return q 

@nb.njit
def objective_dividend(b_next, k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid):
    q = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
    div = z * k**alpha + q * b_next - b - cf  
    return -div

@nb.njit
def compute_exit_decision_step(exit, r, cf, P, k_grid, b_grid):
    """ 
    For a guess on the exit decision function, update the exit decision function by 
        - check if setting b_next = b_max yields positive profits
        - if not, check if an interior solution yields dividend profits (posible if diminishing b_next increases q by enough)
        - if not, the firm exits
    """

    div_max = np.zeros((N_z, N_b, N_k))

    for iz in range(N_z):
        z = z_grid[iz]
        for ik in range(N_k):
            k = k_grid[ik]
            for ib in range(N_b):
                b = b_grid[ib]

                b_max = nu * (1-delta) * k
                k_next = (1-delta) * k_grid[ik]

                # Check dividends when b_next = b_max 
                div_b_max = -objective_dividend(b_max, k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid)

                # If b_next = b_max not feasible, check for an interior solution
                if (div_b_max < 0):
                    b_next = optimizer(objective_dividend, b_grid[0], b_max, args=(k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid))
                    div_max[iz,ib,ik] = -objective_dividend(b_next, k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid)  
                else:
                    b_next = b_max 
                    div_max[iz,ib,ik] = div_b_max

    exit_new = np.asarray(div_max < 0, dtype=np.float64)
    return exit_new

def compute_exit_decision(r, cf, P, k_grid, b_grid):
    """ 
    Iteratively update the exit decision function until convergence
    """
    exit = np.zeros((N_z, N_b, N_k)) 
    error = 1 
    tol = 1e-4
    while error > tol:
        exit_new = compute_exit_decision_step(exit, r, cf, P, k_grid, b_grid)
        error = np.mean(np.abs(exit_new - exit))
        print(error)
        exit = exit_new
    return exit

def compute_q_matrix(exit, r, P, k_grid, b_grid):
    """ 
    Given an exit decision function, compute the price of debt for all choices of k_next, b_next, z 
    """
    q_mat = np.zeros((N_z, N_b, N_k))
    for iz in range(N_z):
        for ik in range(N_k):
            for ib in range(N_b):
                k_next = k_grid[ik]
                b_next = b_grid[ib]
                q_mat[iz,ib,ik] = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
    return q_mat 


@nb.njit 
def compute_b_min(exit, r, P, k_grid, b_grid):
    """ 
    Compute the minimum level of b_next that yields positive dividends:
        - check dividends when b_next = 0. If positive, stops
        - if negative, check for an iterior solution by using a root finder (b_next such that dividends = 0)
        - to use the bisection, we need to find an upper bound on b_next that yields positive dividends. We check first
        b_next = nu * k_next. If this is negative, we take an interior solution using an optimizer. 

    """
    b_min_mat = np.zeros((N_z, N_b, N_k))
    for iz in range(N_z):
        z = z_grid[iz]
        for ik in range(N_k):
            k = k_grid[ik]
            for ib in range(N_b):
                b = b_grid[ib] 
                k_next = (1-delta) * k
                b_max = nu * k_next

                if exit[iz,ib,ik] == 1:
                    continue
                
                div_b_min = -objective_dividend(b_grid[0], k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid)
                if div_b_min < 0:
                    div_b_max = -objective_dividend(b_max, k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid)
                    if div_b_max < 0:
                        b_max = optimizer(objective_dividend, b_grid[0], b_max, args=(k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid))
                    res = qe.optimize.root_finding.bisect(objective_dividend, b_grid[0], b_max, args=(k_next, exit, z, b, k, iz, r, cf, P, k_grid, b_grid))
                    b_min = res.root
                else:
                    b_min = b_grid[0]
                b_min_mat[iz,ib,ik] = b_min
    return b_min_mat

@nb.njit 
def objective_k_prime(k_next, b_next, exit, z, b, k, iz, r, delta, alpha, cf, psi, xi, P, k_grid, b_grid):
    coh = z * k**alpha - b + (1-delta) * k
    q = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi)
    return coh + b_next * q - adj_cost - cf - k_next 

@nb.njit
def objective_k_max(x, exit, z, b, k, iz, r, delta, alpha, cf, psi, xi, P, k_grid, b_grid):
    
    b_next, k_next = x
    coh = z * k**alpha - b + (1-delta) * k
    q = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi)
    div = coh + b_next * q - adj_cost - cf - k_next 
    penalty = 0.0
    if b_next > nu * k_next:
        penalty += -1e4 * np.abs(b_next - nu * k_next)
    if div < 0.0:
        penalty += -1e4 * np.abs(div)
    
    return k_next + penalty

exit = compute_exit_decision(r, cf, P, k_grid, b_grid)
q_mat = compute_q_matrix(exit, r, P, k_grid, b_grid)

@nb.njit
def grid_search_k_max(z, b, k, iz, alpha, exit, r, psi, xi, cf, delta, P, k_grid, b_grid, N_b_next = 100, N_k_next = 100):

    k_choice = np.linspace((1-delta)*k, k_grid[-1], N_k_next)

    for k_next in k_choice:
        b_choice = np.linspace(b, nu * k_next, N_b_next)
        for b_next in b_choice:
            coh = z * k**alpha - b + (1-delta) * k
            q = debt_price_function(iz, k_next, b_next, r, exit, P, k_grid, b_grid)
            adj_cost = compute_adjustment_cost(k_next, k, psi, xi)
            div = coh + b_next * q - adj_cost - cf - k_next 
            if div >= 0.0:
                k_max = k_next
                b_opt = b_next
            if q == 0.0:
                break 

    return k_max, b_opt 

@nb.njit(parallel=True)
def compute_k_max(exit, r, P, k_grid, b_grid, psi, xi, alpha, cf, delta):
    k_max_mat = np.zeros((N_z, N_b, N_k))
    b_opt_k_max = np.zeros((N_z, N_b, N_k))
    for iz in nb.prange(N_z):
        z = z_grid[iz]
        for ik in range(N_k):
            k = k_grid[ik]
            for ib in range(N_b):
                b = b_grid[ib] 
                if exit[iz,ib,ik] == 1:
                    continue

                k_next, b_next = grid_search_k_max(z, b, k, iz, alpha, exit, r, psi, xi, cf, delta, P, k_grid, b_grid)
                #x0 = np.array([b_next,k_next])    
                #res = qe.optimize.nelder_mead(objective_k_max, x0, args=(exit, z, b, k, iz, r, delta, alpha, cf, psi, xi, P, k_grid, b_grid))
                #if res.success == False:
                #    print('Optimization failed when iz = ', iz, 'ib = ', ib, 'ik = ', ik)
                #b_next, k_next = res.x 
                k_max_mat[iz,ib,ik] = k_next
                b_opt_k_max[iz,ib,ik] = b_next
    return k_max_mat, b_opt_k_max

k_max_adj, b_opt_k_max = compute_k_max(exit, r, P, k_grid, b_grid, psi, xi, alpha, cf, delta)
""" 
Things to compute out of the loop:
- exit decision
- b_min when not investing 
- b_min when investing (assuming k_next = (1-delta) * k)
- k_max when adjusting
"""