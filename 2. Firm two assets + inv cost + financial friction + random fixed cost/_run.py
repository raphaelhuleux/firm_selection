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
#model_vfi.prepare()
#model_vfi.solve()

# NVFI 
model_nvfi = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'solve': 'nvfi_analytical'})   

start = time.perf_counter()  # high-resolution timer
model_nvfi.prepare()
model_nvfi.solve()
end = time.perf_counter()
print("Elapsed time:", end - start)


with jit(model_nvfi) as model:
    par = model.par
    sol = model.sol

""" 
Plot policy function
"""
# NVFI
par = model_nvfi.par
sol = model_nvfi.sol

k_policy = sol.k_policy
b_policy = sol.b_policy
div_policy = sol.div_policy
V = sol.V 

k_grid = par.k_grid

plt.plot(k_grid, ((1-sol.exit_policy)*b_policy)[0,0,:].T, label = 'b_policy')
plt.plot(k_grid, ((1-sol.exit_policy)*k_policy)[0,0,:].T- (1-par.delta)*k_grid, label = 'k_policy')
plt.plot(k_grid, ((1-sol.exit_policy)*div_policy)[0,0,:].T, label = 'div_policy')

plt.plot(k_grid, b_policy[-1,0,:].T, label = 'b_policy', linestyle = ':', color = 'C0')
plt.plot(k_grid, k_policy[-1,0,:].T- (1-par.delta)*k_grid, label = 'k_policy', linestyle = ':', color = 'C1')
plt.plot(k_grid, div_policy[-1,0,:].T, label = 'div_policy', linestyle = ':', color = 'C2')
plt.legend()
plt.show()


""" 
Compare grid-search and NVFI
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

ib_plots = np.arange(0, par_nvfi.Nb, 5)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

# Plot capital policy function
for ib in ib_plots:
    axes[0].plot(k_grid, k_policy_vfi[ib, :], label="grid search", linestyle="solid")
    axes[0].plot(k_grid, k_policy_nvfi[ib, :], label="nvfi", linestyle="dashed", color = 'black')

axes[0].set_xlabel("Current Capital (k)")
axes[0].set_ylabel("Next Period Capital (k')")
axes[0].set_title(f"Capital Policy Function (z = {z_grid[iz]:.2f})")

# Plot debt policy function
for ib in ib_plots:
    axes[1].plot(k_grid, b_policy_vfi[ib, :], label="grid search", linestyle="solid")
    axes[1].plot(k_grid, b_policy_nvfi[ib, :], label="nvfi", linestyle="dashed", color = 'black')

axes[1].set_xlabel("Current Capital (k)")
axes[1].set_ylabel("Next Period Debt (b')")
axes[1].set_title(f"Debt Policy Function (z = {z_grid[iz]:.2f})")

# Add a single legend for both subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2)  

plt.tight_layout()
plt.show()