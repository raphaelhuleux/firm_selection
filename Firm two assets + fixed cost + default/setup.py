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
N_k = 50
N_b = 40
N_z = 2

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
def obj_div(k_next, b_next, b, k, z, alpha, delta, r, psi, xi, cf):
    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    div = coh - k_next + b_next - cf

    return div

div_max = np.zeros((N_z, N_b, N_k))
k_max_vec = np.zeros((N_z, N_b, N_k))
for iz in range(N_z):
    z = z_grid[iz]
    for ik in range(N_k):
        k = k_grid[ik]
        for ib in range(N_b):
            b = b_grid[ib]
            b_next = nu * k_grid[ik] 
            k_next = (1-delta) * k_grid[ik]
            div_max[iz,ib,ik] =  z * k**alpha + b_next - b * (1+r) - cf       

np.min(div_max)

exit = div_max < 0
