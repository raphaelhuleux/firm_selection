import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from EconModel import jit
from HeterogenousFirmsModel import HeterogenousFirmsModelClass
import time

# Grid search
model_vfi = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'solve': 'grid_search'})
model_vfi.prepare()
model_vfi.solve()

# NVFI 
model_nvfi = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'Nz': 7, 'Nomega':10, 'omega_sigma':0.5, 'solve': 'nvfi_analytical'})   

start = time.perf_counter()  # high-resolution timer
model_nvfi.prepare()
model_nvfi.solve()
end = time.perf_counter()
print("Elapsed time:", end - start)


with jit(model_nvfi) as model:
    par = model.par
    sol = model.sol
# Check results

# Compare policy functions for nvfi and vfi
par = model_vfi.par
sol = model_vfi.sol

k_policy = sol.k_policy
b_policy = sol.b_policy
div_policy = sol.div_policy
V = sol.V 

k_grid = par.k_grid
b_grid = par.b_grid
z_grid = par.z_grid

plt.plot(k_grid, k_policy[:,0,:].T - (1-par.delta)*k_grid[:,np.newaxis], label = 'k_policy')
plt.plot(k_grid, b_policy[:,0,:].T, label = 'b_policy')
plt.legend()
plt.show()

plt.plot(k_grid, ((1-sol.exit_policy)*b_policy)[0,0,:].T, label = 'b_policy')
plt.plot(k_grid, ((1-sol.exit_policy)*k_policy)[0,0,:].T- (1-par.delta)*k_grid, label = 'k_policy')
plt.plot(k_grid, ((1-sol.exit_policy)*div_policy)[0,0,:].T, label = 'div_policy')

plt.plot(k_grid, b_policy[-1,0,:].T, label = 'b_policy', linestyle = ':', color = 'C0')
plt.plot(k_grid, k_policy[-1,0,:].T- (1-par.delta)*k_grid, label = 'k_policy', linestyle = ':', color = 'C1')
plt.plot(k_grid, div_policy[-1,0,:].T, label = 'div_policy', linestyle = ':', color = 'C2')
plt.legend()
plt.xlabel('k')
plt.show()

# NVFI
par = model_nvfi.par
sol = model_nvfi.sol

k_policy = sol.k_policy
b_policy = sol.b_policy
div_policy = sol.div_policy
V = sol.V 

plt.plot(k_grid, ((1-sol.exit_policy)*b_policy)[0,0,:].T, label = 'b_policy')
plt.plot(k_grid, ((1-sol.exit_policy)*k_policy)[0,0,:].T- (1-par.delta)*k_grid, label = 'k_policy')
plt.plot(k_grid, ((1-sol.exit_policy)*div_policy)[0,0,:].T, label = 'div_policy')

plt.plot(k_grid, b_policy[-1,0,:].T, label = 'b_policy', linestyle = ':', color = 'C0')
plt.plot(k_grid, k_policy[-1,0,:].T- (1-par.delta)*k_grid, label = 'k_policy', linestyle = ':', color = 'C1')
plt.plot(k_grid, div_policy[-1,0,:].T, label = 'div_policy', linestyle = ':', color = 'C2')
plt.legend()
plt.show()

""" 
TODO:
- simulate the lifepath of a firm
- check our results vs Ottonello & Winberry 
- compute the transition matrix of firms
- find out how to add new firms when old ones die 
- add accumulation of cash?
- Comparison point: model with one asset with net worth (and hence no big firms with high debt and a lot of risk)
"""

""" 
with jit(model_vfi) as model:
    par = model.par 
    sol = model.sol
"""

# Extract parameters and solutions for both models
par_vfi, sol_vfi = model_vfi.par, model_vfi.sol
par_nvfi, sol_nvfi = model_nvfi.par, model_nvfi.sol

# Ensure that both grids are the same
assert np.allclose(par_vfi.k_grid, par_nvfi.k_grid), "k_grids do not match!"
assert np.allclose(par_vfi.b_grid, par_nvfi.b_grid), "b_grids do not match!"

# Extract grids
k_grid = par_vfi.k_grid
b_grid = par_vfi.b_grid
z_grid = par_vfi.z_grid

# Choose a specific productivity level for visualization
iz = 0  # Choose the lowest productivity level (modify as needed)

# Extract policy functions for both methods
k_policy_vfi = sol_vfi.k_policy[iz, :, :]
b_policy_vfi = sol_vfi.b_policy[iz, :, :]

k_policy_nvfi = sol_nvfi.k_policy[iz, :, :]
b_policy_nvfi = sol_nvfi.b_policy[iz, :, :]

# Plot capital policy function
plt.figure(figsize=(10, 5))
for ib in range(len(b_grid)):
    plt.plot(k_grid, k_policy_vfi[ib, :], label=f'VFI, b={b_grid[ib]:.2f}', linestyle="solid", alpha=0.5)
    plt.plot(k_grid, k_policy_nvfi[ib, :], label=f'NVFI, b={b_grid[ib]:.2f}', linestyle="dashed", alpha=0.5)

plt.xlabel("Current Capital (k)")
plt.ylabel("Next Period Capital (k')")
plt.title(f"Capital Policy Function Comparison (z = {z_grid[iz]:.2f})")
plt.legend()
plt.show()

# Plot debt policy function
plt.figure(figsize=(10, 5))
for ib in range(len(b_grid)):
    plt.plot(k_grid, b_policy_vfi[ib, :], label=f'VFI, b={b_grid[ib]:.2f}', linestyle="solid", alpha=0.5)
    plt.plot(k_grid, b_policy_nvfi[ib, :], label=f'NVFI, b={b_grid[ib]:.2f}', linestyle="dashed", alpha=0.5)

plt.xlabel("Current Capital (k)")
plt.ylabel("Next Period Debt (b')")
plt.title(f"Debt Policy Function Comparison (z = {z_grid[iz]:.2f})")
plt.legend()
plt.show()