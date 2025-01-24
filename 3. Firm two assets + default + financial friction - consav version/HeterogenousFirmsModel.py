from EconModel import EconModelClass, jit
import numpy as np
import numba as nb
import quantecon as qe
from model_functions import * 
from precompute import compute_exit_decision, compute_q_matrix, compute_b_min, compute_k_max
from vfi_grid_search import solve_vfi_grid_search
from nvfi_analytical import solve_nvfi_analytical

class HeterogenousFirmsModelClass(EconModelClass):
    
    def settings(self): # required
        """ choose settings """
                    
        self.namespaces = ['par', 'sol', 'sim', 'algo'] # must be numba-able
    
    def setup(self): # required
        """ set free parameters """
        
        par = self.par
        algo = self.algo

        # Parameters
        par.alpha = 1/3 # capital share
        par.beta = 1/(1+0.05) # discount factor
        par.delta = 0.1 # depreciation rate

        par.rho = 0.8 # AR(1) shock
        par.sigma_z = 0.2 # std. dev. of shock
        par.psi = 0.05 # convex adjustment cost
        par.xi = 0.0001 # fixed adjustment cost
        par.cf = 0.1 # fixed cost

        par.nu = 100 #  0.9 # leverage ratio
        par.r = (1/par.beta - 1) * 1.1

        # Steady state
        par.z_bar = 1
        par.kbar = (par.alpha * par.z_bar /(1/par.beta-1+par.delta))**(1/(1-par.alpha))

        # Grid
        par.Nk = 40
        par.Nb = 50
        par.Nz = 3

        par.Nk_choice = 100
        par.Nb_choice = 100

        par.k_min = 0.0
        par.k_max = 2*par.kbar

        par.b_min = 0
        par.b_max = min(par.nu*par.k_max, par.k_max) 

        # Algo 
        algo.tol = 1e-6
        algo.howard = True 
        algo.solve = 'grid_search'

    def allocate(self): # required
        """ set compound parameters and allocate arrays """
        
        par = self.par
        sol = self.sol

        # Create grids
        par.k_grid = np.linspace(par.k_min,par.k_max,par.Nk)
        par.b_grid = np.linspace(par.b_min,par.b_max,par.Nb)
        shock = qe.rouwenhorst(par.Nz, par.rho, par.sigma_z)
        par.P = shock.P
        par.z_grid = par.z_bar * np.exp(shock.state_values)
        
        # Create solution arrays
        sol.exit_policy = np.zeros((par.Nz, par.Nb, par.Nk))
        sol.q = np.zeros((par.Nz, par.Nb, par.Nk))
        sol.b_min_keep = np.zeros((par.Nz, par.Nb, par.Nk))
        sol.k_max_adj = np.zeros((par.Nz, par.Nb, par.Nk))

        sol.inaction = np.zeros((par.Nz, par.Nb, par.Nk))
        sol.div_policy = np.zeros((par.Nz, par.Nb, par.Nk))

        sol.b_policy = np.zeros((par.Nz, par.Nb, par.Nk))
        sol.k_policy = np.zeros((par.Nz, par.Nb, par.Nk))
        sol.V = np.zeros((par.Nz, par.Nb, par.Nk))
    
    def prepare(self): # required
        """ precompute specific arrays before solving the model"""
        with jit(self) as model:
            par = model.par
            sol = model.sol
            compute_exit_decision(par, sol)
            compute_q_matrix(par, sol)
            compute_b_min(par, sol)
            compute_k_max(par, sol)

    def solve(self): # user-defined
        """ solve the model """
        with jit(self) as model:
            par = model.par
            sol = model.sol
            if model.algo.solve == 'grid_search':
                solve_vfi_grid_search(par, sol)
            elif model.algo.solve == 'nvfi_analytical':
                solve_nvfi_analytical(par, sol)
