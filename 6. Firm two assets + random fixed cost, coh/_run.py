import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from EconModel import jit
from HeterogenousFirmsModel import HeterogenousFirmsModelClass

# Grid search
#model_vfi = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'solve': 'grid_search', 'Nz': 3})
#model_vfi.prepare()
#model_vfi.solve()

# NVFI 
model_nvfi = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel', par = {'Nz': 7, 'Nomega':10, 'omega_sigma':0.5, 'solve': 'nvfi_analytical'})   
model_nvfi.prepare()
plt.plot(model_nvfi.par.b_grid, model_nvfi.sol.q[0,:,30])
model_nvfi.solve()

""" 
sometimes, no adjust means positive profit but as soon as you pay the fixed cost of investing, profits become negative
we need to account for that by computing another policy
"""
with jit(model_nvfi) as model:
    par = model.par
    sol = model.sol
# Check results

# Compare policy functions for nvfi and vfi
par = model_nvfi.par
sol = model_nvfi.sol

k_policy = sol.k_policy
b_policy = sol.b_policy
div_policy = sol.div_policy
V = sol.V 

k_grid = par.k_grid
m_grid = par.m_grid
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