import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_1d import interp_1d_vec
from consav.linear_interp_1d import interp_1d
from sequence_jacobian import grids, interpolate
from numba import njit, prange
from consav.golden_section_search import optimizer
from scipy.interpolate import interp1d
from scipy import optimize 

# Parameters

alpha = 1/2 # capital share
beta = 0.95 # discount factor
delta = 0.1

rho = 0.8
sigma_z = 0.2
psi = 0.05
xi = 0.01
cf = 1.1

# Steady state
z_bar = 0.9
kbar = (alpha * z_bar /(1/beta-1+delta))**(1/(1-alpha))

# Grid
N_k = 500
N_z = 5
k_grid = np.linspace(0.1,3*kbar,N_k)

shock = qe.rouwenhorst(N_z, rho, sigma_z)
P = shock.P
z_grid = z_bar * np.exp(shock.state_values)

def dividend(k_next, k, z):
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = z * k**alpha - adj_cost - k_next + (1-delta) * k
    return div

kmax_vec = np.zeros((N_z, N_k))

for iz in range(N_z):
    for ik in range(N_k):
        res = optimize.root(dividend, k_grid[ik], args = (k_grid[ik], z_grid[iz]))
        if res.success == False:
            print('error at z = ', iz, 'k = ', ik)
            break
        kmax_vec[iz, ik] = res.x[0]

"""
VFI 
"""

@njit
def bellman_invest(k_next, ik, iz, alpha, delta, psi, xi, cf, k_grid, z_grid, W):
    
    k = k_grid[ik]
    z = z_grid[iz]
    
    coh = z * k**alpha + (1-delta) * k
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next 
    V = div + np.interp(k_next, k_grid, W) - cf
    return -V 


@njit
def bellman_inaction(ik, iz, alpha, delta, cf, k_grid, z_grid, W):
    
    k = k_grid[ik]
    z = z_grid[iz]

    k_next = (1-delta) * k
    div = z * k**alpha 
    V = div + np.interp(k_next, k_grid, W) - cf
    return V 
    
@njit
def vfi_step(V, Vexit, psi, xi, delta, alpha, cf, z_grid, k_grid):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    
    W = beta * P @ V
    for iz in range(N_z):
        for ik in range(N_k):
            k = k_grid[ik]
            #kmax = k_grid[-1]
            kmax = kmax_vec[iz, ik]
            k_opt = optimizer(bellman_invest, (1-delta)*k, kmax, args = (ik, iz, alpha, delta, psi, xi, cf, k_grid, z_grid, W[iz,:]))

            Vinv = -bellman_invest(k_opt, ik, iz, alpha, delta, psi, xi, cf, k_grid, z_grid, W[iz,:])
            Vina = bellman_inaction(ik, iz, alpha, delta, cf, k_grid, z_grid, W[iz,:])
            
            if Vinv >= Vina:
                k_policy[iz, ik] = k_opt
                V_new[iz, ik] = Vinv
            else:
                k_policy[iz, ik] = (1-delta) * k_grid[ik]
                V_new[iz, ik] = Vina

            if V_new[iz, ik] < Vexit[iz, ik]:
                V_new[iz, ik] = Vexit[iz, ik]
                k_policy[iz, ik] = 0
                
    return V_new, k_policy

def vfi(V_init, Vexit, psi, xi, delta, alpha, cf, z_grid, k_grid):
    error = 1
    tol = 1e-5
    V = V_init.copy()
    while error > tol:
        Vnew, k_policy = vfi_step(V, Vexit, psi, xi, delta, alpha, cf, z_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
    return V, k_policy



"""
EGM
"""


@njit
def upperenv_vec(W, k_endo, z_grid, k_grid, alpha, psi, xi, delta):
    """Interpolate value function and consumption to exogenous grid."""
    n_z, n_a = W.shape
    k = np.zeros_like(k_endo)
    V = -np.inf * np.ones_like(k_endo)

    # loop over other states, collapsed into single axis
    
    for iz in range(n_z):
            # loop over segments of endogenous asset grid from EGM (not necessarily increasing)
        for ja in range(n_a - 1):
            k_low, k_high = k_endo[iz, ja], k_endo[iz, ja + 1]
            # a_endo[ib,ja] tells you which level a today makes you willing to 
            # end up in a_grid[ja] and receive utility u(c) + W[ib, ja]
            # c = a_endo[ib,ja] * (1+r) + z[ib] - a_grid[ja]
            W_low, W_high = W[iz, ja], W[iz, ja + 1]
            kp_low, kp_high = k_grid[ja], k_grid[ja + 1]
            
        # loop over exogenous asset grid (increasing) 
            for ia in range(n_a):  
                kcur = k_grid[ia]
                
                """  
                For every other state ib, we check 
                - if for every segment a_endo[ib,ja] (a_low) and a_endo[ib,j+1] (a_high)
                we check if there is a point on the exogenous grid a_grid that lies within
                this segment
                - if this is the case, we compute c0, v0 on the exogenous grid (by simple interpolation), 
                and update the policy and value function if th
                
                """
                interp = (k_low <= kcur <= k_high) 
                extrap = (ja == n_a - 2) and (kcur > k_endo[iz, n_a - 1])

                # exploit that a_grid is increasing
                if (k_high < kcur < k_endo[iz, n_a - 1]):
                    break

                if interp or extrap:
                    # @njit
                    # def interpolate_point(x, x0, x1, y0, y1):
                    # y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    # return y
                    W0 = interpolate.interpolate_point(kcur, k_low, k_high, W_low, W_high)
                    # interpolate continuation value
                    k0 = interpolate.interpolate_point(kcur, k_low, k_high, kp_low, kp_high)
                    # interpolate policy function
                    adj_cost = psi / 2 * ((k0 - (1-delta)*kcur)  / kcur)**2 * kcur + xi * kcur 
                    div = z_grid[iz] * kcur**alpha - k0 + (1-delta) * kcur - adj_cost 
                    V0 = div + W0

                    # upper envelope, update if new is better
                    if V0 > V[iz, ia]:
                        k[iz, ia] = k0 
                        V[iz, ia] = V0

    for iz in range(n_z):
        for ik in range(n_a):
            if k[iz, ik] < (1-delta) * k_grid[ik]:
                k[iz, ik] = (1-delta) * k_grid[ik]
                adj_cost = xi * k_grid[ik] 

                div = z_grid[iz] * k_grid[ik]**alpha - k[iz,ik] + (1-delta) * k_grid[ik] - adj_cost
                V[iz, ik] = div + interp_1d(k_grid, W[iz,:], k[iz, ik])

            if k[iz,ik] > kmax_vec[iz,ik]:
                k[iz,ik] = kmax_vec[iz,ik]
                adj_cost = psi / 2 * ((k[iz,ik] - (1-delta)*k_grid[ik])  / k_grid[ik])**2 * k_grid[ik] + xi * k_grid[ik] 
                div = z_grid[iz] * k_grid[ik]**alpha - k[iz,ik] + (1-delta) * k_grid[ik] - adj_cost
                V[iz,ik] = div + interp_1d(k_grid, W[iz,:], k[iz,ik])

    return V, k

@njit
def egm_step(V, Va, Vexit, k_grid, z_grid, alpha, psi, xi, delta, beta, cf, P):    
    # 1. Compute post decision value function W and q 
    W = beta * P @ V 
    q = beta * P @ Va
            
    # 2. Invert the foc to get the policy function
    k_endo = psi / (q + psi * (1-delta) - 1) * (k_grid[np.newaxis,:])                               
            
    # 3. Apply the upper-envelope 
    Vinv, kinv = upperenv_vec(W, k_endo, z_grid, k_grid, alpha, psi, xi, delta)
    
    # 4. Compute the value function for inaction
    Vina = np.empty_like(V)
    temp = np.zeros((W.shape[1]))
    for iz in range(N_z):
        interp_1d_vec(k_grid, W[iz,:], (1-delta)*k_grid, temp)
        Vina[iz,:] = z_grid[iz] * k_grid**alpha + temp 

    # 5. Take the max and update the policy function    
    V_new = np.maximum(Vinv - cf, Vina - cf)
    k_policy = np.where(Vinv >= Vina, kinv, (1 - delta) * k_grid[np.newaxis, :])

    k_policy = np.where(Vexit > V_new, 0, k_policy)
    V_new = np.maximum(V_new, Vexit)

    # 6. Update the marginal post decision value q
    Va_new = np.empty_like(V)
    Va_new[..., 1:-1] = (V_new[..., 2:] - V_new[..., :-2]) / (k_grid[2:] - k_grid[:-2])
    Va_new[..., 0] = (V_new[..., 1] - V_new[..., 0]) / (k_grid[1] - k_grid[0])
    Va_new[..., -1] = (V_new[..., -1] - V_new[..., -2]) / (k_grid[-1] - k_grid[-2])

    return V_new, Va_new, k_policy

            
def egm(V, Va, Vexit, k_grid, z_grid, alpha, psi, xi, delta, beta, cf, P):
    error = 1
    tol = 1e-6
    while error > tol:
        Vnew, Va_new, k_policy = egm_step(V, Va, Vexit, k_grid, z_grid, alpha, psi, xi, delta, beta, cf, P)
        
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        Va = Va_new
    return V, Va, k_policy

""" 
Code
"""

Vexit = z_grid[:,np.newaxis] * k_grid[np.newaxis,:]**alpha + (1-delta) * k_grid[np.newaxis,:] - (psi/2)* (delta-1)**2 * k_grid[np.newaxis,:] - xi*k_grid[np.newaxis,:]  


k_init = k_grid**(1/2)
V_init = (z_grid[:, np.newaxis] * k_init**alpha) * 1/(1-beta)
Va_init = np.empty_like(V_init)
Va_init[..., 1:-1] = (V_init[..., 2:] - V_init[..., :-2]) / (k_grid[2:] - k_grid[:-2])
Va_init[..., 0] = (V_init[..., 1] - V_init[..., 0]) / (k_grid[1] - k_grid[0])
Va_init[..., -1] = (V_init[..., -1] - V_init[..., -2]) / (k_grid[-1] - k_grid[-2])


V_vfi, k_policy_vfi = vfi(V_init, Vexit, psi, xi, delta, alpha, cf, z_grid, k_grid)

Va_vfi = np.empty_like(V_init)
Va_vfi[..., 1:-1] = (V_vfi[..., 2:] - V_vfi[..., :-2]) / (k_grid[2:] - k_grid[:-2])
Va_vfi[..., 0] = (V_vfi[..., 1] - V_vfi[..., 0]) / (k_grid[1] - k_grid[0])
Va_vfi[..., -1] = (V_vfi[..., -1] - V_vfi[..., -2]) / (k_grid[-1] - k_grid[-2])

V_egm, Va_egm, k_policy_egm = egm(V_init, Va_init, Vexit, k_grid, z_grid, alpha, psi, xi, delta, beta, cf, P)

plt.plot(k_grid, k_policy_egm.T - (1-delta) * k_grid[:,np.newaxis])
plt.plot(k_grid, k_policy_vfi.T - (1-delta) * k_grid[:,np.newaxis], linestyle = ':')
plt.show()

plt.plot(k_grid, Va_egm.T)
plt.plot(k_grid, Va_vfi.T, linestyle = ':')
plt.show()
div = z_grid[:, np.newaxis] * k_grid**alpha - k_policy_vfi + (1-delta) * k_grid - psi / 2 * ((k_policy_vfi - (1-delta)*k_grid)  / k_grid)**2 * k_grid - xi * k_grid        


plt.plot(k_grid, V_vfi.T)
plt.plot(k_grid, V_egm.T, linestyle = ':')
plt.plot(k_grid, Vexit.T, color = 'black')