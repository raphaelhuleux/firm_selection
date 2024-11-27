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
def bellman_invest(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W):
    z = z_grid[iz]

    coh = z * k**alpha + (1-delta) * k - b * (1+r)
    adj_cost = psi / 2 * (k_next - (1-delta)*k)**2 / k + xi * k 
    div = coh - adj_cost - k_next + b_next - cf

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(b_grid,k_grid, W[iz], b_next, k_next)
    
    return V

@njit 
def bellman_inaction(b_next, b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W):
    z = z_grid[iz]
    k_next = (1-delta) * k

    coh = z * k**alpha - b * (1+r)
    div = coh + b_next - cf

    if div < 0:
        return -np.inf 
    
    V = div + interp_2d(b_grid,k_grid, W[iz], b_next, k_next)
    
    return V


@njit
def grid_search_invest(b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W, Nb_choice = 100, Nk_choice = 100):

    Vmax = -np.inf 
    b_min = b * (1+r) - z_grid[iz] * k**alpha - cf
    if b_min < b_grid[0]:
        b_min = b_grid[0]
    b_max = nu * k
    b_choice = np.linspace(b_min, b_max, Nb_choice)
    
    k_min = (1-delta) * k + 1e-8
    k_max = k_grid[-1]
    k_choice = np.linspace(k_min, k_max, Nk_choice)

    for ik_next in range(Nk_choice):
        k_next = k_choice[ik_next]
        for ib_next in range(Nb_choice):
            b_next = b_choice[ib_next]
            V = bellman_invest(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
            if V > Vmax:
                Vmax = V
                b_max = b_next
                k_max = k_next

    return Vmax, b_max, k_max

@njit
def grid_search_inaction(b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W, Nb_choice = 100):

    Vmax = -np.inf 
    b_min = b * (1+r) - z_grid[iz] * k**alpha 
    if b_min < b_grid[0]:
        b_min = b_grid[0]
    b_max = nu * k
    b_choice = np.linspace(b_min, b_max, Nb_choice)
    
    for ib_next in range(Nb_choice):
        b_next = b_choice[ib_next]
        V = bellman_inaction(b_next, b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W)
        if V > Vmax:
            Vmax = V
            b_max = b_next
    
    return Vmax, b_max


@njit(parallel = True)
def vfi_step(V, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):
    V_new = np.empty_like(V)
    k_policy = np.empty_like(V)
    b_policy = np.empty_like(V)

    W = beta * fast_expectation(P, V)

    for iz in prange(N_z):
        for ik in range(N_k):
            for ib in range(N_b):

                if exit[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                    k_policy[iz, ib, ik] = 0
                    b_policy[iz, ib, ik] = 0
                else:
                    b = b_grid[ib]
                    k = k_grid[ik]

                    Vinv, b_inv, k_inv = grid_search_invest(b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
                    Vina, b_ina = grid_search_inaction(b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W)

                    if Vinv > Vina:
                        V_new[iz, ib, ik] = Vinv
                        k_policy[iz, ib, ik] = k_inv
                        b_policy[iz, ib, ik] = b_inv
                    else:   
                        V_new[iz, ib, ik] = Vina
                        k_policy[iz, ib, ik] = (1-delta) * k
                        b_policy[iz, ib, ik] = b_ina

              
    return V_new, k_policy, b_policy


@njit(parallel = True)
def howard_step(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    V_new = np.empty_like(W)
    for iz in prange(N_z):
        for ik in range(N_k):
            for ib in range(N_b):
                if exit[iz,ib, ik]:
                    V_new[iz, ib, ik] = 0
                else:
                    b = b_grid[ib]
                    k = k_grid[ik]
                    b_next = b_policy[iz, ib, ik]
                    k_next = k_policy[iz, ib, ik]
                    
                    if k_next == (1-delta) * k:
                        V_new[iz, ib, ik] = bellman_inaction(b_next, b, k, iz, alpha, delta, r, cf, z_grid, b_grid, k_grid, W)
                    else:
                        V_new[iz, ib, ik] = bellman_invest(b_next, k_next, b, k, iz, alpha, delta, psi, xi, r, cf, z_grid, b_grid, k_grid, W)
                    
    return V_new

def howard(V, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid):

    for n in range(50):
        W = beta * fast_expectation(P, V)
        V = howard_step(W, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)

    return V

def vfi(V_init, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid, tol = 1e-5, do_howard = True):
    error = 1

    V = V_init.copy()
    while error > tol:
        Vnew, k_policy, b_policy = vfi_step(V, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        if do_howard:
            V = howard(V, k_policy, b_policy, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid)
    return V, k_policy, b_policy 

def _simulate(b0, k0, iz0, k_policy, b_policy, P, z_grid, b_grid, k_grid, T = 100):

    nZ = len(z_grid)
    b = np.zeros((T))
    k = np.zeros((T))
    inv = np.zeros((T))
    div = np.zeros((T))
    adj_cost = np.zeros((T))
    ex = np.zeros((T), dtype = np.int32)
    iz = np.zeros((T), dtype = np.int32)
    z = np.zeros((T), dtype = np.int32)


    b[0] = b0
    k[0] = k0   
    iz[0] = iz0

    for t in range(T):
        if t < T-1:
            iz[t+1] = np.random.choice(np.arange(nZ), p=P[iz[t]])
        z[t] = z_grid[iz[t]]
        k[t+1] = interp_2d(b_grid,k_grid, k_policy[iz[t]], b[t], k[t])
        b[t+1] = interp_2d(b_grid,k_grid, b_policy[iz[t]], b[t], k[t])
        inv[t] = k[t+1] - (1-delta) * k[t]
        if inv[t] > 0:
            adj_cost[t] = psi / 2 * (k[t+1] - (1-delta) * k[t])**2 / k[t] + xi * k[t]
        div[t] = z[t] * k[t]**alpha + (1-delta) * k[t] - b[t] * (1+r) - adj_cost[t] - k[t+1] + b[t+1] - cf 
        if div[t] < 0:  
            ex[t:] = 1
            break 

    return np.array([b, k, ex, div, z, inv, adj_cost, iz])

def simulate(k_policy, b_policy, P, z_grid, b_grid, k_grid, T = 1000, N = 10_000):

    sims = np.zeros((N, 8, T))
    for i in range(N):
        k0 = k_grid[-1] #0.1 #np.random.uniform(k_min, k_max)
        b0 = 0
        iz0 = 1 #np.random.choice(N_z)

        sims[i] = _simulate(b0, k0, iz0, k_policy, b_policy, P, z_grid, b_grid, k_grid, T = T)

    return sims

k_policy = np.ones((N_z, N_b, N_k)) * k_grid[np.newaxis,np.newaxis,:]
b_policy = np.ones((N_z, N_b, N_k)) * b_grid[np.newaxis,:,np.newaxis]

V_init = np.zeros((N_z, N_b, N_k))
V, k_policy, b_policy = vfi(V_init, psi, xi, delta, alpha, cf, r, z_grid, b_grid, k_grid, do_howard = True)

sim = simulate(0, 0.1, 0, k_policy, b_policy, P, z_grid, b_grid, k_grid, T = 100)
plt.plot(k_grid, k_policy[0,0,:], label = 'k_policy')
plt.plot(k_grid, b_policy[0,0,:], label = 'b_policy')

plt.legend()
plt.xlabel('k')
plt.title('Policy function for iz = 0, ib  0')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(k_grid, (1-exit[0,0,:])*(k_policy[0, 0, :] - (1 - delta) * k_grid), label='Net investment with low TFP')
axes[0].plot(k_grid, (1-exit[-1,0,:])*(k_policy[-1, 0, :] - (1 - delta) * k_grid), label='Net investment with high TFP')
axes[0].legend()
axes[0].set_xlabel('k')
axes[0].set_title('Net investment in the absence of debt')
axes[1].plot(k_grid, (1-exit[0,-10,:])*(k_policy[0, -10, :] - (1 - delta) * k_grid), label='Net investment with low TFP')
axes[1].plot(k_grid, (1-exit[-1,-10,:])*(k_policy[-1, -10, :] - (1 - delta) * k_grid), label='Net investment with high TFP')
axes[1].legend()
axes[1].set_xlabel('k')
axes[1].set_title('Net investment when high debt')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(b_grid,(1- exit[0,:,0])*(b_policy[0, :, 0]-b_grid), label='Debt choice with low TFP')
axes[0].plot(b_grid, (1-exit[-1,:,0])*(b_policy[-1, :, 0]-b_grid), label='Debt choice with high TFP')
axes[0].legend()
axes[0].set_xlabel('debt')
axes[0].set_ylabel('net dissaving')
axes[0].set_title('Debt choice with low capital')

axes[1].plot(b_grid, (1- exit[0,:,-1])*(b_policy[0, :, -1]-b_grid), label='Debt choice with low TFP')
axes[1].plot(b_grid, (1- exit[-1,:,-1])*(b_policy[-1, :, -1]-b_grid), label='Debt choice with high TFP')
axes[1].legend()
axes[1].set_xlabel('debt')
axes[1].set_title('Debt choice with high capital')
plt.tight_layout()
plt.show()



coh = z_grid[:,np.newaxis,np.newaxis] * k_grid[np.newaxis,np.newaxis,:]**alpha + (1-delta) * k_grid[np.newaxis,np.newaxis,:] - b_grid[np.newaxis,:,np.newaxis] * (1+r) 
adj_cost = psi / 2 * (k_policy - (1-delta) * k_grid[np.newaxis,np.newaxis,:])*2 / k_grid[np.newaxis,np.newaxis,:] + xi * k_grid[np.newaxis,np.newaxis,:]
div_opt = (1-exit) * (coh - adj_cost - k_policy + b_policy - cf )

B_grid, K_grid = np.meshgrid(b_grid, k_grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B_grid, K_grid, exit[0,:,:].T, cmap='viridis', edgecolor='k')

ax.set_xlabel('Debt')
ax.set_ylabel('Capital')
ax.set_zlabel('Exit')
plt.title("3D Plot of a Binary Variable")
plt.show()
#Exit (1) when debt is high and capital is low, the non-negativity constraint on dividend cannot hold
