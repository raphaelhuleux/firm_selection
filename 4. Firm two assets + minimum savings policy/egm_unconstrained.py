import numpy as np 
from consav.linear_interp_1d import interp_1d_vec
from consav.linear_interp_1d import interp_1d
from sequence_jacobian import interpolate
from model_functions import compute_adjustment_cost
from numba import njit

"""
EGM
"""

@njit
def upperenv_vec(W, k_endo, par):

    z_grid = par.z_grid 
    k_grid = par.k_grid
    alpha = par.alpha
    xi = par.xi
    delta = par.delta
    psi = par.psi

    """Interpolate value function and consumption to exogenous grid."""
    Nz, n_a = W.shape
    k = np.zeros_like(k_endo)
    V = -np.inf * np.ones_like(k_endo)

    # loop over other states, collapsed into single axis
    
    for iz in range(Nz):
            # loop over segments of endogenous asset grid from EGM (not necessarily increasing)
        for ja in range(n_a - 1):
            k_low, k_high = k_endo[iz, ja], k_endo[iz, ja + 1]
            W_low, W_high = W[iz, ja], W[iz, ja + 1]
            kp_low, kp_high = k_grid[ja], k_grid[ja + 1]
            
        # loop over exogenous asset grid (increasing) 
            for ia in range(n_a):  
                kcur = k_grid[ia]
                
                interp = (k_low <= kcur <= k_high) 
                extrap = (ja == n_a - 2) and (kcur > k_endo[iz, n_a - 1])

                # exploit that a_grid is increasing
                if (k_high < kcur < k_endo[iz, n_a - 1]):
                    break

                if interp or extrap:
                    W0 = interpolate.interpolate_point(kcur, k_low, k_high, W_low, W_high)
                    # interpolate continuation value
                    k0 = interpolate.interpolate_point(kcur, k_low, k_high, kp_low, kp_high)
                    # interpolate policy function
                    adj_cost = compute_adjustment_cost(k0, kcur, delta, psi, xi)
                    #adj_cost = psi / 2 * ((k0 - (1-delta)*kcur)  / kcur)**2 * kcur + xi * kcur 
                    div = z_grid[iz] * kcur**alpha - k0 + (1-delta) * kcur - adj_cost 
                    V0 = div + W0

                    # upper envelope, update if new is better
                    if V0 > V[iz, ia]:
                        k[iz, ia] = k0 
                        V[iz, ia] = V0

    for iz in range(Nz):
        for ik in range(n_a):
            if k[iz, ik] < (1-delta) * k_grid[ik]:
                k[iz, ik] = (1-delta) * k_grid[ik]
                adj_cost = xi * k_grid[ik] 

                div = z_grid[iz] * k_grid[ik]**alpha - k[iz,ik] + (1-delta) * k_grid[ik] - adj_cost
                V[iz, ik] = div + interp_1d(k_grid, W[iz,:], k[iz, ik])

    return V, k

@njit   
def compute_expectation_unconstrained(V, Va, par):

    # Expectation with respect to z and omega
    W = par.beta * par.P @ V - par.cf
    Wa = par.beta * par.P @ Va 

    # Expectation with respect to capital shocks
    W_new = np.zeros_like(W)
    Wa_new = np.zeros_like(Wa)

    for iz in range(par.Nz):
        for ik in range(par.Nk):
            k_next_chosen = par.k_grid[ik]
            V_temp = 0.0
            Va_temp = 0.0
            for i_shock in range(par.Nkshock): 
                k_next = k_next_chosen * par.k_shock_grid[i_shock]
                V_temp += par.k_shock_p[i_shock] * interp_1d(par.k_grid, W[iz,:], k_next)
                Va_temp += par.k_shock_p[i_shock] * interp_1d(par.k_grid, Wa[iz,:], k_next)
            W_new[iz,ik] = V_temp
            Wa_new[iz,ik] = Va_temp

    return W_new, Wa_new

        
@njit
def egm_step(V, Va, par): 
    # 1. Compute post decision value function W and q 
    #W = par.beta * par.P @ V 
    #q = par.beta * par.P @ Va
            
    W, Wa = compute_expectation_unconstrained(V, Va, par)

    # 2. Invert the foc to get the policy function
    k_endo = par.psi / (Wa + par.psi * (1-par.delta) - 1) * (par.k_grid[np.newaxis,:])                               
            
    # 3. Apply the upper-envelope 
    Vinv, kinv = upperenv_vec(W, k_endo, par)
    
    # 4. Compute the value function for inaction
    Vina = np.empty_like(V)
    temp = np.zeros((W.shape[1]))
    for iz in range(par.Nz):
        interp_1d_vec(par.k_grid, W[iz,:], (1-par.delta)*par.k_grid, temp)
        Vina[iz,:] = par.z_grid[iz] * par.k_grid**par.alpha + temp 

    # 5. Take the max and update the policy function    
    V_new = np.maximum(Vinv, Vina)
    k_policy = np.where(Vinv >= Vina, kinv, (1 - par.delta) * par.k_grid[np.newaxis, :])

    # 6. Update the marginal post decision value q
    Va_new = np.empty_like(V)
    Va_new[..., 1:-1] = (V_new[..., 2:] - V_new[..., :-2]) / (par.k_grid[2:] - par.k_grid[:-2])
    Va_new[..., 0] = (V_new[..., 1] - V_new[..., 0]) / (par.k_grid[1] - par.k_grid[0])
    Va_new[..., -1] = (V_new[..., -1] - V_new[..., -2]) / (par.k_grid[-1] - par.k_grid[-2])

    return V_new, Va_new, k_policy

            
def egm_unconstrained_ss(ss, par):
    error = 1
    tol = 1e-6

    k_init = par.k_grid**(1/2)
    V = (par.z_grid[:, np.newaxis] * k_init**par.alpha) * 1/(1-par.beta)
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (par.k_grid[2:] - par.k_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (par.k_grid[1] - par.k_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (par.k_grid[-1] - par.k_grid[-2])

    print('Obtaining unconstrained capital policy with EGM')
    print('-----------------------------------------------')
    while error > tol:
        Vnew, Va_new, k_policy = egm_step(V, Va, par)
        
        error = np.sum(np.abs(Vnew - V))
        print(error)
        V = Vnew
        Va = Va_new
    print('Done sweetie')
    print(' ')

    return k_policy 


""" 
Code
"""

def obtain_minimum_savings_policy(k_policy, par):
    b_policy = np.zeros((par.Nz, par.Nk))
    b_policy_new = np.zeros((par.Nz, par.Nk, par.Nz))

    tol = 1e-4
    error = 1 

    print('Obtaining minimum savings policy')
    print('--------------------------------')

    while error > tol:
        b_policy_new = obtain_minimum_savings_policy_step(k_policy, b_policy, par)
        b_policy_new = np.min(b_policy_new, axis = 2)
        error = np.max(np.abs(b_policy_new - b_policy))
        print(error)
        b_policy = b_policy_new.copy()
        b_policy_new = np.zeros((par.Nz, par.Nk, par.Nz))

    print('Done sweetie')
    print(' ')
    return b_policy

@njit
def obtain_minimum_savings_policy_step(k_policy, b_policy, par):
    k_grid = par.k_grid
    z_grid = par.z_grid
    alpha = par.alpha
    delta = par.delta
    psi = par.psi
    xi = par.xi

    b_policy_new = np.zeros((par.Nz, par.Nk, par.Nz))
    for iz in range(par.Nz):
        for ik in range(par.Nk):
            for iz_next in range(par.Nz):
                z_next = z_grid[iz_next]
                k_next_chosen = k_policy[iz, ik]
                k_next = k_next_chosen * par.k_shock_grid[0] # worst case scenario

                k_next2 = interp_1d(k_grid, k_policy[iz_next,:], k_next)
                b_next2 = interp_1d(k_grid, b_policy[iz_next,:], k_next)
                b_next2 = np.maximum(np.minimum(b_next2, par.nu * k_next2), par.b_grid[0])

                if (1-delta) * k_next < k_next2:
                    adj_cost = psi / 2 * ((k_next2 - (1-delta)*k_next)  / k_next)**2 * k_next + xi * k_next
                else:
                    adj_cost = 0
                y = z_next * k_next**alpha 

                # Worst case scenario for fixed cost
                b_pol = y + (1-delta) * k_next + np.minimum(- k_next2 - adj_cost + par.beta * b_next2, 0) - par.omega_grid[-1]
                b_policy_new[iz, ik, iz_next] = np.maximum(np.minimum(b_pol, par.nu * k_next), par.b_grid[0])

    return b_policy_new

@njit
def get_unconstrained_indicator(k_policy, b_policy, par):
    unconstrained_indicator = np.zeros((par.Nz, par.Nb, par.Nk))
    div = np.zeros((par.Nz, par.Nb, par.Nk))

    for iz in range(par.Nz):
        z = par.z_grid[iz]
        for ik in range(par.Nk):
            k = par.k_grid[ik]
            for ib in range(par.Nb):
                b = par.b_grid[ib]
                y = z * k**par.alpha

                k_next = k_policy[iz,ik]
                b_next = b_policy[iz,ik]
                adj_cost = compute_adjustment_cost(k_next, k, par.delta, par.psi, par.xi)
                div[iz,ib,ik] = y - k_next + (1-par.delta) * k - adj_cost - b + par.beta * b_next 

                if div[iz,ib,ik] < 0:
                    unconstrained_indicator[iz,ib,ik] = 0
                else:
                    unconstrained_indicator[iz,ib,ik] = 1

    return unconstrained_indicator, div

if __name__ == "__main__":
    import numpy as np 
    import matplotlib.pyplot as plt
    from EconModel import jit
    from HeterogenousFirmsModel import HeterogenousFirmsModelClass

    model = HeterogenousFirmsModelClass(name = 'HeterogenousFirmsModel')
    with jit(model) as model:
        par = model.par

    V, Va, k_policy = egm(V_init, Va_init, par)
    b_policy = obtain_minimum_savings_policy(k_policy, par)
    unconstrained_indicator = get_unconstrained_indicator(k_policy, b_policy, par)

