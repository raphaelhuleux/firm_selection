from EconModel import EconModelClass, jit
import numpy as np
import numba as nb
import quantecon as qe
from model_functions import * 
from precompute import compute_exit_decision, compute_q_matrix, compute_b_min, compute_k_max
from vfi_grid_search import solve_vfi_grid_search

class HeterogenousFirmsModelClass(EconModelClass):
    
    def settings(self): # required
        """ choose settings """
                    
        self.namespaces = ['par','sol'] # must be numba-able
    
    def setup(self): # required
        """ set free parameters """
        
        par = self.par
        
        par.alpha = 1/2 # capital share
        par.beta = 0.95 # discount factor
        par.delta = 0.1

        par.rho = 0.8
        par.sigma_z = 0.2
        par.psi = 0.05
        par.xi = 0.001
        par.cf = 0.1

        par.nu = 0.9
        par.r = (1/par.beta - 1) * 1.1

        # Steady state
        par.z_bar = 1
        par.kbar = (par.alpha * par.z_bar /(1/par.beta-1+par.delta))**(1/(1-par.alpha))

        # Grid
        par.N_k = 60
        par.N_b = 80
        par.N_z = 3

        par.k_min = 0.0
        par.k_max = 2*par.kbar

        par.b_min = 0
        par.b_max = par.nu*par.k_max


    def allocate(self): # required
        """ set compound parameters and allocate arrays """
        
        par = self.par
        sol = self.sol

        # Create grids
        par.k_grid = np.linspace(par.k_min,par.k_max,par.N_k)
        par.b_grid = np.linspace(par.b_min,par.b_max,par.N_b)
        shock = qe.rouwenhorst(par.N_z, par.rho, par.sigma_z)
        par.P = shock.P
        par.z_grid = par.z_bar * np.exp(shock.state_values)
        
        # Create solution arrays
        sol.exit_policy = np.zeros((par.N_z, par.N_b, par.N_k))
        sol.q = np.zeros((par.N_z, par.N_b, par.N_k))
        sol.b_min_keep = np.zeros((par.N_z, par.N_b, par.N_k))
        sol.k_max_adj = np.zeros((par.N_z, par.N_b, par.N_k))

        sol.inaction = np.zeros((par.N_z, par.N_b, par.N_k))
        sol.dividends = np.zeros((par.N_z, par.N_b, par.N_k))
        
        sol.b_policy = np.zeros((par.N_z, par.N_b, par.N_k))
        sol.k_policy = np.zeros((par.N_z, par.N_b, par.N_k))
        sol.V = np.zeros((par.N_z, par.N_b, par.N_k))
    
    def prepare(self): # required
        with jit(self) as model:
            par = model.par
            sol = model.sol
            compute_exit_decision(par, sol)
            compute_q_matrix(par, sol)
            compute_b_min(par, sol)
            compute_k_max(par, sol)

    def solve(self): # user-defined
        """ solve the model"""
        with jit(self) as model:
            par = model.par
            sol = model.sol
            solve_vfi_grid_search(par, sol)
