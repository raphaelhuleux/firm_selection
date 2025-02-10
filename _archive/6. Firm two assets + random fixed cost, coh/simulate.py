import numpy as np 
from consav.linear_interp_2d import interp_2d

def _simulate(b0, k0, iz0, k_policy, b_policy, delta, alpha, xi, cf, psi, r, P, z_grid, b_grid, k_grid, T = 100):

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

def simulate(k_policy, b_policy, delta, alpha, xi, cf, psi, r, P, z_grid, b_grid, k_grid, T = 1000, N = 10_000):

    sims = np.zeros((N, 8, T))
    for i in range(N):
        k0 = k_grid[-1] #0.1 #np.random.uniform(k_min, k_max)
        b0 = 0
        iz0 = 1 #np.random.choice(Nz)

        sims[i] = _simulate(b0, k0, iz0, k_policy, b_policy, delta, alpha, xi, cf, psi, r, P, z_grid, b_grid, k_grid, T = T)

    return sims