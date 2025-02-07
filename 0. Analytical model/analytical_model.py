import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite
from scipy.optimize import minimize_scalar
from scipy import optimize

# Permanent productivity
z_min = 0.8
z_max = 2
nZ = 10
z_grid = np.linspace(z_min, z_max, nZ)

# Shock
mu = 0.0       # Mean of ln(epsilon)
sigma = 0.3    # Standard deviation of ln(epsilon)

nE = 100
nodes, weights = roots_hermite(nE)
e_grid = np.exp(np.sqrt(2) * sigma * nodes + mu)
e_prob = weights / np.sqrt(np.pi)
e_cdf = np.cumsum(e_prob)

# Precompute cumulative sums for interpolation
profits_cumsum_base = np.cumsum(e_prob * e_grid)
profits_cumsum_base = profits_cumsum_base  # Ensure it's an array

# Parameters
mu = 0.95
w = 1
alpha = 2/3
n = 0.01
r = 1.05
z = 1 

def default_probability_obj(epsilon_bar, l, z, alpha, w, n, r): 
    d = w * l - n 
    r_tilde = (z * epsilon_bar * l**alpha) / d 
    default_probability = np.interp(epsilon_bar, e_grid, e_cdf)
    expected_profits = l**alpha * np.interp(epsilon_bar, e_grid, profits_cumsum_base)
    bank_error = (1 - default_probability) * r_tilde * d + (1-mu) * default_probability * expected_profits - r * d
    return bank_error

def profits(l, z, alpha, w, n, r):
    d = w * l - n 
    res = optimize.root(default_probability_obj, 0.5, args=(l, z, alpha, w, n, r), method='hybr')
    if res.success == False: 
        return np.inf 
    epsilon_bar = res.x[0]
    r_tilde = (z * epsilon_bar * l**alpha) / d 
    profits_temp = z * e_grid * l**alpha - r_tilde * d
    profits = np.sum(e_prob * np.maximum(profits_temp, 0))
    return -profits

def find_max_l(z, alpha, w, n, r):
    l_grid = np.linspace(0.05, 10, 1000)
    profits_grid = profits_vec(l_grid, z, alpha, w, n, r)
    
    # Find the index of the first value that is equal to inf
    inf_index = np.where(np.isinf(profits_grid))[0]
    if len(inf_index) > 0:
        l_max = l_grid[inf_index[0]]  # Use the last valid value before inf
    
    l_grid = np.linspace(0.05, l_max, 1000)
    return l_grid

profits_vec = np.vectorize(profits, excluded=['z', 'alpha', 'w', 'n', 'r'])

l_grid = find_max_l(z, alpha, w, n, r)
profits_grid = profits_vec(l_grid, z, alpha, w, n, r)
plt.plot(l_grid, -profits_grid)
plt.xlabel('Labor Input (l)')
plt.ylabel('Profit')
plt.title('Profit Function of the Firm')
plt.show()

def get_eq_default_prob(z, alpha, w, n, r):
    
    l_grid = find_max_l(z, alpha, w, n, r)
    profits_grid = profits_vec(l_grid, z, alpha, w, n, r)
    l_opt = np.argmin(profits_grid)
    l = l_grid[l_opt]
    
    res = minimize_scalar(profits, method='golden', bracket = (l*0.9,l*1.1), args=(z, alpha, w, n, r)) 
    if res.success == False: 
        print("Didn't find an optimum for z  =", z)
    l = res.x
    
    res = optimize.root(default_probability_obj, 0.5, args=(l, z, alpha, w, n, r), method='hybr') 
    if res.success == False: 
        print("Didn't find an equilibrium epsilon_bar for z  =", z)
    epsilon_bar = res.x[0]
    
    d = w * l - n 
    r_tilde = (z * epsilon_bar * l**alpha) / d 
    default_probability = np.interp(epsilon_bar, e_grid, e_cdf)
    
    return r_tilde, default_probability, l, -profits(l, z, alpha, w, n, r)

# Low interest rate
r = 1.03
r_tilde_low = np.zeros((nZ))
default_probability_low = np.zeros((nZ))
l_opt_low = np.zeros((nZ))
profits_eq_low = np.zeros((nZ))
for iz, z in enumerate(z_grid):
    r_tilde_low[iz], default_probability_low[iz], l_opt_low[iz], profits_eq_low[iz] = get_eq_default_prob(z, alpha, w, n, r)

# High interest rate
r = 1.05
r_tilde_high = np.zeros((nZ))
default_probability_high = np.zeros((nZ))
l_opt_high = np.zeros((nZ))
profits_eq_high = np.zeros((nZ))
for iz, z in enumerate(z_grid):
    r_tilde_high[iz], default_probability_high[iz], l_opt_high[iz], profits_eq_high[iz] = get_eq_default_prob(z, alpha, w, n, r)


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Firm Equilibrium Results')

axs[0, 0].plot(z_grid, default_probability_low, label = 'r = 0.03')
axs[0, 0].plot(z_grid, default_probability_high, label = 'r = 0.05')
axs[0, 0].set_title('Default Probability')
axs[0, 0].set_xlabel('Productivity (z)')
axs[0, 0].set_ylabel('Probability')
axs[0, 0].legend()

axs[0, 1].plot(z_grid, (r_tilde_low - 1.03)*100)
axs[0, 1].plot(z_grid, (r_tilde_high - 1.05)*100)
axs[0, 1].set_title('Interest Rate Premium')
axs[0, 1].set_xlabel('Productivity (z)')
axs[0, 1].set_ylabel('%')

axs[1, 0].plot(z_grid, l_opt_low)
axs[1, 0].plot(z_grid, l_opt_high)
axs[1, 0].set_title('Labor Input')
axs[1, 0].set_xlabel('Productivity (z)')
axs[1, 0].set_ylabel('Labor (l)')

axs[1, 1].plot(z_grid, profits_eq_low)
axs[1, 1].plot(z_grid, profits_eq_high)
axs[1, 1].set_title('Equilibrium Profits')
axs[1, 1].set_xlabel('Productivity (z)')
axs[1, 1].set_ylabel('Profits')

plt.tight_layout()
plt.show()
