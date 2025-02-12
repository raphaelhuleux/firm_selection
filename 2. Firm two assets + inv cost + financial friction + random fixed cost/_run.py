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

# NVFI - analytical
model_analytical = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'solve_b': 'analytical', 'howard': True})   

model_analytical.prepare()
model_analytical.solve_steady_state()
model_analytical.solve_transition()

with jit(model_analytical) as model:
    par = model.par
    ss = model.ss
    trans = model.trans

ss_K = np.sum(ss.D * ss.k_policy)
K = np.sum(trans.D * trans.k_policy, axis = (1,2,3))
q = np.sum(trans.D[1:] * trans.q[:-1], axis = (1,2,3))
plt.plot(K[1:])
plt.plot(q)

plt.plot(par.b_grid, np.sum(ss.D, axis = (0,2)))
plt.xlabel('b')
plt.ylabel('Density')
plt.show()

plt.plot(par.k_grid, np.sum(ss.D, axis = (0,1)))
plt.xlabel('k')
plt.ylabel('Density')
plt.show()

# NVFI - optimizer
model_optimizer = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'solve_b': 'optimizer', 'howard': True})   

model_optimizer.prepare()
model_optimizer.solve_firm_problem()
model_optimizer.compute_steady_state_distribution()
with jit(model_optimizer) as model:
    par = model.par
    sol = model.sol

plt.plot(par.b_grid, np.sum(ss.D, axis = (0,2)))
plt.xlabel('b')
plt.ylabel('Density')
plt.show()

plt.plot(par.k_grid, np.sum(ss.D, axis = (0,1)))
plt.xlabel('k')
plt.ylabel('Density')
plt.show()

""" 
Plot policy function
"""

# NVFI
par = model_optimizer.par
sol = model_optimizer.sol

k_policy = ss.k_policy
b_policy = ss.b_policy
div_policy = ss.div_policy
V = ss.V 

k_grid = par.k_grid

plt.plot(k_grid, ((1-ss.exit_policy)*b_policy)[0,0,:].T, label = 'b_policy')
plt.plot(k_grid, ((1-ss.exit_policy)*k_policy)[0,0,:].T- (1-par.delta)*k_grid, label = 'k_policy')
plt.plot(k_grid, ((1-ss.exit_policy)*div_policy)[0,0,:].T, label = 'div_policy')

plt.plot(k_grid, b_policy[-1,0,:].T, label = 'b_policy', linestyle = ':', color = 'C0')
plt.plot(k_grid, k_policy[-1,0,:].T- (1-par.delta)*k_grid, label = 'k_policy', linestyle = ':', color = 'C1')
plt.plot(k_grid, div_policy[-1,0,:].T, label = 'div_policy', linestyle = ':', color = 'C2')
plt.legend()
plt.show()

""" 
Compare grid-search and NVFI
"""

# Extract parameters and solutions for both models
par_analytical, sol_vfi = model_analytical.par, model_analytical.sol
par_optimizer, sol_optimizer = model_optimizer.par, model_optimizer.sol

# Ensure that both grids are the same
assert np.allclose(par_analytical.k_grid, par_optimizer.k_grid), "k_grids do not match!"
assert np.allclose(par_analytical.b_grid, par_optimizer.b_grid), "b_grids do not match!"

# Extract grids
k_grid = par_analytical.k_grid
b_grid = par_analytical.b_grid
z_grid = par_analytical.z_grid

# Choose a specific productivity level for visualization
iz = 0  # Choose the lowest productivity level (modify as needed)

# Extract policy functions for both methods
k_policy_vfi = sol_vfi.k_policy[iz, :, :]
b_policy_vfi = sol_vfi.b_policy[iz, :, :]

k_policy_nvfi = sol_optimizer.k_policy[iz, :, :]
b_policy_nvfi = sol_optimizer.b_policy[iz, :, :]

ib_plots = np.arange(0, par_optimizer.Nb, 5)

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