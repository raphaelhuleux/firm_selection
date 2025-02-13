from EconModel import EconModelClass, jit
import numpy as np
import numba as nb
import quantecon as qe
from model_functions import * 
from precompute import compute_exit_decision_ss, compute_exit_decision_trans, compute_b_min, compute_k_max, compute_exit_decision_adj
from vfi_grid_search import solve_vfi_grid_search
from nvfi import solve_nvfi_ss, solve_problem_firm_trans
from consav.quadrature import log_normal_gauss_hermite
from consav.grids import nonlinspace # grids
from compute_distribution import distribution_ss, distribution_trans

class HeterogenousFirmsModelClass(EconModelClass):
    
    def settings(self): # required
        """ choose settings """
                    
        self.namespaces = ['par', 'ss', 'trans'] # must be numba-able
    
    def setup(self): # required
        """ set free parameters """
        
        par = self.par
        trans = self.trans

        # Parameters
        par.alpha = 0.21 # capital share
        par.beta = 0.99 # discount factor
        par.delta = 0.025 # depreciation rate

        par.rho = 0.9 # AR(1) shock
        par.sigma_z = 0.03 # std. dev. of shock
        par.omega_sigma = 0.5
        par.psi = 0.05 # convex adjustment cost
        par.xi = 0.0001 # fixed adjustment cost
        par.cf = 0.1 # fixed cost

        par.r = (1/par.beta - 1) * 1.1

        # Steady state
        par.z_bar = 1
        par.kbar = (par.alpha * par.z_bar /(1/par.beta-1+par.delta))**(1/(1-par.alpha))

        par.T = 150 

        # r shock 
        par.rho_r = 0.61
        par.sigma_r = 0.01/4
        trans.r = par.r + par.sigma_r * par.rho_r **(np.arange(par.T))

        # Grid
        par.Nk = 80
        par.Nb = 70
        par.Nz = 6
        par.Nomega = 7

        par.Nk_choice = 100
        par.Nb_choice = 100

        par.k_min = 0.0
        par.k_max = 3*par.kbar

        par.b_min = 0
        par.b_max = par.k_max / 3  

        # Algo 
        par.tol = 1e-6
        par.howard = True 
        par.solve = 'nvfi'
        par.solve_b = 'analytical'

    def allocate(self): # required
        """ set compound parameters and allocate arrays """
        
        par = self.par
        ss = self.ss
        trans = self.trans 

        ss.r = par.r
        # Create grids
        #par.k_grid =  nonlinspace(par.k_min,par.k_max,par.Nk,1.1)
        #par.b_grid =  nonlinspace(par.b_min,par.b_max,par.Nb,1.1)

        par.k_grid = np.linspace(par.k_min,par.k_max,par.Nk)
        par.b_grid = np.linspace(par.b_min,par.b_max,par.Nb)

        shock = qe.rouwenhorst(par.Nz, par.rho, par.sigma_z)
        par.P = shock.P
        par.z_grid = par.z_bar * np.exp(shock.state_values)
        
        par.omega_grid, par.omega_p = log_normal_gauss_hermite(par.omega_sigma, n=par.Nomega,mu=par.cf)
                
        # Create ssution arrays
        ss.exit_policy = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.exit_policy_adj = np.zeros((par.Nz, par.Nb, par.Nk))

        ss.q = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.b_min_keep = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.k_max_adj = np.zeros((par.Nz, par.Nb, par.Nk))

        ss.inaction = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.div_policy = np.zeros((par.Nz, par.Nb, par.Nk))

        ss.b_policy = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.k_policy = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.V = np.zeros((par.Nz, par.Nb, par.Nk))
        ss.D = np.zeros((par.Nz, par.Nb, par.Nk))

        # Trans
        trans.q = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
        trans.exit_policy = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
        trans.exit_policy_adj = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
        trans.b_policy = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
        trans.k_policy = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
        trans.V = np.zeros((par.T, par.Nz, par.Nb, par.Nk))
        trans.D = np.zeros((par.T, par.Nz, par.Nb, par.Nk))

    def prepare(self): # required
        """ precompute specific arrays before ssving the model"""
        with jit(self) as model:
            par = model.par
            ss = model.ss

            ss.exit_policy[...], ss.exit_policy_adj[...], ss.q[...] = compute_exit_decision_ss(ss.r, par)
            ss.b_min_keep[...] = compute_b_min(ss.q, ss.exit_policy, par)
            ss.k_max_adj[...] = compute_k_max(ss.q, ss.exit_policy, par)

    def solve_steady_state(self):
        
        with jit(self) as model:
            par = model.par
            ss = model.ss

            if model.par.solve == 'grid_search':
                solve_vfi_grid_search(par, ss)
            elif model.par.solve == 'nvfi':
                solve_nvfi_ss(ss, par)

            distribution_ss(ss, par)

    def solve_transition(self):
        with jit(self) as model:
            par = model.par
            ss = model.ss
            trans = model.trans

            trans.exit_policy[...], trans.exit_policy_adj[...], trans.q[...] = compute_exit_decision_trans(trans.r, ss, par)
            solve_problem_firm_trans(trans, ss, par)
            distribution_trans(trans, ss, par)
            
