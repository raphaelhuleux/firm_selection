import numpy as np 
import matplotlib.pyplot as plt
from EconModel import jit
from HeterogenousFirmsModel import HeterogenousFirmsModelClass

""" 
TODO data
- get French data 
- 

TODO model
- Change reversibility of capital
- Add firm discount future profits by the interest rate
- Compute life cycle of a firm
- Solve problem of the firm without real frictions
- Add capital quality shocks 
- Add checks that constraints are met in policy functions
- Add checks that constraints are met in transition
- Add growth or trends so that big firms keep investing and having debt?
"""

# NVFI - analytical
model = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'solve_b': 'analytical', 'howard': True, 'iter_howard':50})   
model.prepare()
model.solve_steady_state()
model.solve_transition()

with jit(model) as model:
    par = model.par
    ss = model.ss
    trans = model.trans

np.sum(trans.exit_policy * trans.D)
plt.plot(par.b_grid, ss.q[0,:,[0,1,2,5,10,15,20]].T)
plt.plot(par.b_grid, np.ones_like(par.b_grid)*(1/(1+ss.r)), linestyle = '--')
plt.xlabel('b')
plt.ylabel('q')
plt.show()

b_policy = ss.b_policy
k_policy = ss.k_policy
print('share debt = ', np.sum(ss.D[:,1:,:]))
print("nu * k' - b' =", np.min(par.nu * k_policy - b_policy))

ss_B = np.sum(ss.D * par.b_grid[None,:,None])
ss_K = np.sum(ss.D * par.k_grid[None,None,:])

np.max((np.abs(ss.q - 1/(1+ss.r))))
ss_q = np.sum(ss.D * ss.q)

share_default = np.sum((ss.exit_policy==1) * ss.D)

share_default_trans = np.sum(trans.exit_policy * trans.D, axis = (1,2,3))
plt.plot(share_default_trans)
plt.show()

K = np.sum(trans.D * par.k_grid[None,None,None,:], axis = (1,2,3))
B = np.sum(trans.D * par.b_grid[None,None,:,None], axis = (1,2,3))
Q = np.sum(trans.D[1:] * trans.q[:-1], axis = (1,2,3))

np.max(np.abs(trans.q - ss.q))

np.max(np.abs(trans.k_policy[-1] - ss.k_policy))

plt.plot(K-ss_K)
plt.ylabel('K')
plt.xlabel('t')
plt.show()

plt.plot(1/Q-1 - (1/ss_q - 1), label = 'Q')
plt.plot(trans.r[1:] - ss.r, label = 'r')
plt.ylabel('r (implied)')
plt.xlabel('t')
plt.legend()
plt.show()

plt.plot(B-ss_B)
plt.ylabel('B')
plt.xlabel('t')
plt.show()

plt.plot(par.b_grid, np.sum(ss.D, axis = (0,2)), label = 'b')
plt.plot(par.k_grid, np.sum(ss.D, axis = (0,1)), label = 'k')
plt.plot(par.b_grid, np.sum(trans.D[-1], axis = (0,2)), linestyle = ':')
plt.plot(par.k_grid, np.sum(trans.D[-1], axis = (0,1)), linestyle = ':')
plt.xlabel('k, b')
plt.ylabel('Density')
plt.legend()
plt.show()

k_grid = par.k_grid
b_grid = par.b_grid

data = np.sum(ss.D, axis = (1))
plt.imshow(data, cmap='viridis', aspect='auto', 
           extent=[b_grid.min(), b_grid.max(), k_grid.min(), k_grid.max()], 
           origin='lower')  # Adjusts coordinates

plt.colorbar()

# Set the actual b_grid and k_grid values as labels
plt.xticks(ticks=np.linspace(b_grid.min(), b_grid.max(), len(b_grid)), labels=np.round(b_grid, 2))
plt.yticks(ticks=np.linspace(k_grid.min(), k_grid.max(), len(k_grid)), labels=np.round(k_grid, 2))

plt.xlabel('b_grid')
plt.ylabel('k_grid')

plt.show()

for iz in range(par.Nz):
    plt.plot(par.b_grid[:10], np.sum(ss.D[iz], axis = (1))[:10] / np.sum(ss.D[iz]), label = 'z = ' + str(par.z_grid[iz]))
plt.xlabel('b')
plt.ylabel('Density')
plt.legend()
plt.show()

for iz in range(par.Nz):
    print('z = ', par.z_grid[iz])
    print('b = ', np.sum(ss.D[iz] * par.b_grid[:,None]) / np.sum(ss.D[iz]))
    print('k = ', np.sum(ss.D[iz] * par.k_grid[None,:]) / np.sum(ss.D[iz]))
    print('prob default = ', np.sum(ss.exit_policy[iz] * ss.D[iz]) / np.sum(ss.D[iz]))
    print(' ')

""" 
Plot policy function
"""

# NVFI
par = model.par
sol = model.ss

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
par, sol = model.par, model.ss

# Extract grids
k_grid = par.k_grid
b_grid = par.b_grid
z_grid = par.z_grid

# Choose a specific productivity level for visualization
iz = 0  # Choose the lowest productivity level (modify as needed)

# Extract policy functions for both methods
k_policy = sol.k_policy[iz, :, :]
b_policy = sol.b_policy[iz, :, :]

ib_plots = np.arange(0, par.Nb, 5)

# Create figure with two scubplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

# Plot capital policy function
for ib in ib_plots:
    axes[0].plot(k_grid, k_policy[ib, :], label="grid search", linestyle="solid")
    axes[0].plot(k_grid, k_policy[ib, :], label="nvfi", linestyle="dashed", color = 'black')

axes[0].set_xlabel("Current Capital (k)")
axes[0].set_ylabel("Next Period Capital (k')")
axes[0].set_title(f"Capital Policy Function (z = {z_grid[iz]:.2f})")

# Plot debt policy function
for ib in ib_plots:
    axes[1].plot(k_grid, b_policy[ib, :], label="grid search", linestyle="solid")
    axes[1].plot(k_grid, b_policy[ib, :], label="nvfi", linestyle="dashed", color = 'black')

axes[1].set_xlabel("Current Capital (k)")
axes[1].set_ylabel("Next Period Debt (b')")
axes[1].set_title(f"Debt Policy Function (z = {z_grid[iz]:.2f})")

# Add a single legend for both subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2)  

plt.tight_layout()
plt.show()