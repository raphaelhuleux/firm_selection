import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_1d import interp_1d_vec
from consav.linear_interp_1d import interp_1d
from consav.linear_interp_2d import interp_2d, interp_2d_vec
from consav.golden_section_search import optimizer
from sequence_jacobian import grids, interpolate
from numba import njit, prange
from scipy.interpolate import interp1d
from scipy import optimize 
from mpl_toolkits.mplot3d import Axes3D
import quantecon as qe 
from scipy.optimize import golden 

# Parameters

alpha = 1/2 # capital share
beta = 0.95 # discount factor
delta = 0.1

rho = 0.8
sigma_z = 0.2
psi = 0.05
xi = 0 #0.01
cf = 0 

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
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf

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
            coh = z * k**alpha + (1-delta) * k - b * (1+r)
            adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
            div_max[iz,ib,ik] = coh - adj_cost - k_next + b_next - cf       

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
def bellman_grid_search(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W):
    z = z_grid[iz]

    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(b_grid,k_grid, W[iz], b_next, k_next)
    
    return V

@njit
def grid_search(b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W,Nb_choice = 150, Nk_choice = 150):

    Vmax = -np.inf 
    b_min = b * (1+r) - z_grid[iz] * k**alpha 
    if b_min < b_grid[0]:
        b_min = b_grid[0]
    b_max = nu * k
    b_choice = np.linspace(b_min, b_max, Nb_choice)
    
    k_min = (1-delta) * k 
    k_max = k_grid[-1]
    k_choice = np.linspace(k_min, k_max, Nk_choice)

    for ik_next in range(Nk_choice):
        k_next = k_choice[ik_next]
        for ib_next in range(Nb_choice):
            b_next = b_choice[ib_next]
            V = bellman_grid_search(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
            if V > Vmax:
                Vmax = V
                b_max = b_next
                k_max = k_next

    return Vmax, b_max, k_max


@njit
def bellman_fun(x, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W):

    b_next, k_next = x
    z = z_grid[iz]

    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf

    if div < 0:
        penalty = div * 1e4
    
    V = div + interp_2d(b_grid, k_grid, W[iz], b_next, k_next) - penalty
    
    return V 


@njit(parallel = True)
def vfi_step(V, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    b_policy = np.empty_like(V)

    W = beta * fast_expectation(P, V)

    for iz in prange(N_z):
        for ik in range(N_k):
            for ib in range(N_b):

                if i_div_neg[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0
                else:
                    b = b_grid[ib]
                    k = k_grid[ik]
                    Vmax, b_max, k_max = grid_search(b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)

                    V_new[iz, ib, ik] = Vmax
                    k_policy[iz, ib, ik] = k_max
                    b_policy[iz, ib, ik] = b_max

                    """ 
                    coh = z_grid[iz] * k**alpha + (1-delta) * k - b * (1+r)
                    k_min = (1-delta) * k_grid[ik]
                    k_max = k_grid[-1]
                    b_min = -coh - cf - (1-delta) * k
                    if b_min < b_grid[0]:
                        b_min = 0
                    b_max = nu * k_grid[ik]

                    bounds = np.array([[b_min,b_max],[k_min,k_max]])
                    
                    init = np.array([b_max, k_max])

                    res = qe.optimize.nelder_mead(bellman_fun, init, bounds = bounds, args = (b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W))

                    k_policy[iz, ib, ik] = res.x[1]
                    b_policy[iz, ib, ik] = res.x[0]
                    V_new[iz, ib, ik] = bellman_fun(res.x, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
                    """
                    
    return V_new, k_policy, b_policy

@njit
def howard_step(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    V_new = np.empty_like(W)
    for iz in range(N_z):
        for ik in range(N_k):
            for ib in range(N_b):
                if i_div_neg[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                else:
                    b = b_grid[ib]
                    k = k_grid[ik]
                    b_next = b_policy[iz, ib, ik]
                    k_next = k_policy[iz, ib, ik]
                    V_new[iz, ib, ik] = bellman_grid_search(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)

    return V_new

def howard(V, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    W = beta * fast_expectation(P, V)

    for n in range(30):
        V = howard_step(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
        W = beta * fast_expectation(P, V)

    return V

def vfi(V_init, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid, tol = 1e-5):
    error = 1

    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = vfi_step(V, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        V = howard(V, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
    return V, k_policy, b_policy 

V_init = np.zeros((N_z, N_b, N_k))
V, k_policy, b_policy = vfi(V_init, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

plt.plot(k_grid, k_policy[0,0,:], label = 'k_policy')
plt.plot(k_grid, b_policy[0,0,:], label = 'b_policy')
plt.legend()
plt.xlabel('k')
plt.title('Policy function for iz = 0, ib  0')
plt.show()

coh = z_grid[:,np.newaxis,np.newaxis] * k_grid[np.newaxis,np.newaxis,:]**alpha + (1-delta) * k_grid[np.newaxis,np.newaxis,:] - b_grid[np.newaxis,:,np.newaxis] * (1+r) 
adj_cost = psi / 2 * (k_policy - (1-delta)*k_grid[np.newaxis,np.newaxis,:])**2 / k_grid[np.newaxis,np.newaxis,:] + xi * k_grid[np.newaxis,np.newaxis,:]
div_opt = (1-i_div_neg) * (coh - adj_cost - k_policy + b_policy - cf )
