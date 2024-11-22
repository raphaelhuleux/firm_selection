import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_1d import interp_1d_vec
from consav.linear_interp_1d import interp_1d
from consav.linear_interp_2d import interp_2d

from sequence_jacobian import grids, interpolate
from numba import njit, prange
from consav.golden_section_search import optimizer
from scipy.interpolate import interp1d
from scipy import optimize 
from mpl_toolkits.mplot3d import Axes3D
import quantecon as qe 

# Parameters

alpha = 1/2 # capital share
beta = 0.95 # discount factor
delta = 0.1

rho = 0.8
sigma_z = 0.2
psi = 0.01
xi = 0 #0.01
cf = 0 

nu = 0.9
r = (1/beta - 1) * 1.1

# Steady state
z_bar = 1
kbar = (alpha * z_bar /(1/beta-1+delta))**(1/(1-alpha))

# Grid
N_k = 60
N_b = 50
N_z = 2

k_min = 0.1
k_max = 3*kbar

b_min = 0
b_max = nu*k_max

b_grid = np.linspace(b_min,b_max,N_b)
k_grid = np.linspace(k_min,k_max,N_k)

shock = qe.rouwenhorst(N_z, rho, sigma_z)
P = shock.P
z_grid = z_bar * np.exp(shock.state_values)


div_max = z_grid[:,np.newaxis,np.newaxis] * k_grid[np.newaxis,np.newaxis,:]**alpha - b_grid[np.newaxis,:,np.newaxis] * (1+r) +  nu * k_grid[np.newaxis,np.newaxis,:] - cf 
np.min(div_max)

i_div_neg = div_max < 0

"""
VFI 
"""


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

@njit
def bellman(ik_next, ib_next, iz, ib, ik, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W):
    
    k = k_grid[ik]
    b = b_grid[ib]
    z = z_grid[iz]

    k_next = k_grid[ik_next]
    b_next = b_grid[ib_next]
    
    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf
    #V = div + interp_2d(b_next, k_next, b_grid, k_grid, W) 
    V = div + W[ib_next,ik_next]

    return V 

@njit
def bellman_fun(x, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W):
    k_next, b_next = x
    z = z_grid[iz]
   
    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf
    V = div + interp_2d(b_grid, k_grid, W, b_next, k_next) 
    
    return -V 

@njit 
def grid_search_opt(iz, ib, ik, alpha, delta, psi, xi, cf, r, z_grid, b_grid, k_grid, W):
    
    k = k_grid[ik]
    
    Vmax = -np.inf
    k_opt = 0
    b_opt = 0


    for ik_next in range(N_k):
        for ib_next in range(N_b):
            b_next = b_grid[ib_next]
            if b_next > nu * k:
                break 

            V = bellman(ik_next, ib_next, iz, ib, ik, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
            if V > Vmax:
                Vmax = V
                k_opt = ik_next
                b_opt = ib_next


    return k_grid[k_opt], b_grid[b_opt], Vmax

@njit
def vfi_step(V, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    b_policy = np.empty_like(V)

    W = beta * fast_expectation(P, V)
    for iz in range(N_z):
        for ik in range(N_k):
            for ib in range(N_b):
                if i_div_neg[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0
                else:
                    k_max, b_max, Vmax = grid_search_opt(iz, ib, ik, alpha, delta, psi, xi, cf, r, z_grid, b_grid, k_grid, W[iz])
                    res = qe.optimize.nelder_mead(bellman_fun, np.array([k_max,b_max]), args=(b_grid[ib], k_grid[ik], iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W[iz]))
                    k_policy[iz, ib, ik] = res.x[0]
                    b_policy[iz, ib, ik] = res.x[1]
                    V_new[iz, ib, ik] = -res.fun
                
    return V_new, k_policy, b_policy

def vfi(V_init, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):
    error = 1
    tol = 1e-5
    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = vfi_step(V, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
    return V, k_policy, b_policy 

V_init = np.zeros((N_z, N_b, N_k))
V, k_policy, b_policy = vfi(V_init, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)




plt.plot(k_grid, k_policy[0,0,:])

plt.plot(k_grid, b_policy[0,30,:])