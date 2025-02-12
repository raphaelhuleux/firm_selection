import numpy as np 
from numba import njit 
from model_functions import fast_expectation

def distribution_ss(par, sol, tol = 1e-8):
    
    b_i, b_pi = get_lottery(sol.b_policy, par.b_grid)
    k_i, k_pi = get_lottery(sol.k_policy, par.k_grid)
    o_i, o_pi = get_lottery(par.omega_grid[:,np.newaxis] + par.b_grid[np.newaxis,:], par.b_grid)

    D = np.zeros_like(sol.b_policy)
    D[-1,0,:] = np.ones((par.Nk)) / par.Nk
    #D[0,0,1] = 1

    for it in range(25_000):
        D_new = forward_policy_2d(D, b_i, b_pi, k_i, k_pi, sol.exit_policy, par)
        D_new = update_distribution_omega(D_new, o_i, o_pi, par) # update omega
        D_new = fast_expectation(par.P.T, D_new)
        entrants = 1-np.sum(D_new)
        D_new[0,0,1] += entrants# / par.Nz
        if it % 10 == 0 and equal_tolerance(D_new, D, tol):
            return D_new
        D = D_new   


def distribution_trans(par, trans):
    pass 

@njit
def equal_tolerance(x1, x2, tol):
    # "ravel" flattens both x1 and x2, without making copies, so we can compare the
    # with a single for loop even if they have multiple dimensions
    x1 = x1.ravel()
    x2 = x2.ravel()

    # iterate over elements and stop immediately if any diff by more than tol
    for i in range(len(x1)):
        if np.abs(x1[i] - x2[i]) >= tol:
            return False
    return True

@njit
def get_lottery(xq, x):
    """Does interpolate_coord_robust where xq must be a vector, more general function is wrapper"""

    shape = xq.shape
    xq = xq.ravel()
    n = len(x)
    nq = len(xq)
    xqi = np.empty(nq, dtype=np.uint32)
    xqpi = np.empty(nq)

    for iq in range(nq):
        if xq[iq] < x[0]:
            ilow = 0
        elif xq[iq] > x[-2]:
            ilow = n-2
        else:
            # start binary search
            # should end with ilow and ihigh exactly 1 apart, bracketing variable
            ihigh = n-1
            ilow = 0
            while ihigh - ilow > 1:
                imid = (ihigh + ilow) // 2
                if xq[iq] > x[imid]:
                    ilow = imid
                else:
                    ihigh = imid

        xqi[iq] = ilow
        xqpi[iq] = (x[ilow+1] - xq[iq]) / (x[ilow+1] - x[ilow])

    return xqi.reshape(shape), xqpi.reshape(shape)

@njit
def forward_policy_2d(D, b_i, b_pi, k_i, k_pi, exit_policy, par):

    Dnew = np.zeros(D.shape)
    nZ, nB, nK = D.shape

    for iz in range(par.Nz):
        for ib in range(par.Nb):
            for ik in range(par.Nk):
                if exit_policy[iz, ib, ik] == 1:
                    Dnew[iz, ib, ik] = 0
                else:
                    ikp = k_i[iz, ib, ik]
                    alpha = k_pi[iz, ib, ik]

                    ibp = b_i[iz, ib, ik]
                    beta = b_pi[iz, ib, ik]
                    Dnew[iz, ibp,     ikp] += alpha     * beta     * D[iz, ib, ik]
                    Dnew[iz, ibp+1,   ikp] += alpha     * (1-beta) * D[iz, ib, ik]
                    Dnew[iz, ibp,   ikp+1] += (1-alpha) * beta     * D[iz, ib, ik]
                    Dnew[iz, ibp+1, ikp+1] += (1-alpha) * (1-beta) * D[iz, ib, ik]

    return Dnew

@njit
def update_distribution_omega(D, o_i, o_pi, par):

    Dnew = np.zeros(D.shape)

    for ib in range(par.Nb):
        for iomega in range(par.Nomega):
            ibp = o_i[iomega, ib]
            alpha = o_pi[iomega, ib]
            p = par.omega_p[iomega]
            Dnew[:, ibp, :] += alpha * p * D[:, ib, :]
            Dnew[:, ibp, :] += (1-alpha) * p * D[:, ib, :]

    return Dnew

