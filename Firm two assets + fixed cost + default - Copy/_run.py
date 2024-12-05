
import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from setup import * 
from vfi import *
from nvfi import * 

k_policy = np.ones((N_z, N_b, N_k)) * k_grid[np.newaxis,np.newaxis,:]
b_policy = np.ones((N_z, N_b, N_k)) * b_grid[np.newaxis,:,np.newaxis]

V_init = np.zeros((N_z, N_b, N_k))
#V_vfi, k_policy_vfi, b_policy_vfi = vfi(V_init, beta, nu, psi, xi, delta, alpha, cf, r, P, z_grid, b_grid, k_grid)
V_nvfi, k_policy_nvfi, b_policy_nvfi = nvfi(V_init, beta, nu, psi, xi, delta, alpha, cf, r, P, z_grid, b_grid, k_grid, tol = 1e-5)

#plt.plot(k_grid, k_policy_vfi[0,0,:] - (1-delta)*k_grid, label = 'vfi')
plt.plot(k_grid, k_policy_nvfi[0,0,:] - (1-delta)*k_grid, label = 'nvfi')

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
