import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 

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
xi = 0.01
cf = 0.1

nu = 0.9
r = (1/beta - 1) * 1.1

# Steady state
z_bar = 1
kbar = (alpha * z_bar /(1/beta-1+delta))**(1/(1-alpha))

# Grid
N_k = 100
N_b = 100
N_z = 5

k_min = 0.1
k_max = 1.2*kbar

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

div_max_keep = np.zeros((N_z, N_b, N_k))
div_max_adj = np.zeros((N_z, N_b, N_k))
k_max_adj = np.zeros((N_z, N_b, N_k))

@nb.njit
def objective_div_max_inv(k_next, z, k, b):
    b_next = nu * k_next
    adj_cost = compute_adjustment_cost(k_next, k, psi, xi) 
    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    div = coh - adj_cost - k_next + b_next - cf
    return div 


for iz in range(N_z):
    z = z_grid[iz]
    for ik in range(N_k):
        k = k_grid[ik]
        for ib in range(N_b):
            b = b_grid[ib]
            b_next = nu * (1-delta) * k_grid[ik] 
            k_next = (1-delta) * k_grid[ik]
            div_max_keep[iz,ib,ik] =  z * k**alpha + b_next - b * (1+r) - cf     
            div_max_adj[iz,ib,ik] = objective_div_max_inv(k_next, z, k, b)

            if div_max_adj[iz,ib,ik] > 0:
                if objective_div_max_inv(k_grid[-1], z, k, b) > 0:
                    k_max_adj[iz,ib,ik] = k_grid[-1]
                else:
                    res = qe.optimize.root_finding.bisect(objective_div_max_inv, (1-delta)*k, k_grid[-1], args = (z, k, b))  
                    k_max_adj[iz,ib,ik] = res.root

exit_keep = div_max_keep < 0
exit_adj = div_max_adj < 0