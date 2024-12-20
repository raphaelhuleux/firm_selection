import numpy as np 
import numba as nb 
import matplotlib.pyplot as plt
import quantecon as qe 
from consav.linear_interp_2d import interp_2d
from numba import njit, prange
import quantecon as qe 
from EconModel import jit
from HeterogenousFirmsModel import HeterogenousFirmsModelClass

model = HeterogenousFirmsModelClass(name='HeterogenousFirmsModel')
model.prepare()
model.solve()

par = model.par
sol = model.sol

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

plt.plot(k_grid, b_policy[0,0,:].T, label = 'b_policy')
plt.plot(k_grid, k_policy[0,0,:].T- (1-par.delta)*k_grid, label = 'k_policy')
plt.plot(k_grid, div_policy[0,0,:].T, label = 'div_policy')
plt.legend()

""" 
with jit(model) as model:
    par = model.par 
    sol = model.sol
"""