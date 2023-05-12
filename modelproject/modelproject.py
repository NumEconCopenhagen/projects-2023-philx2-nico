from scipy import optimize
import numpy as np
import sympy as sm
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tabulate import tabulate
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

class SolowModelClass:
    
    def __init__(self, do_print=True):
        """ create the model """

        # if do_print: print('initializing the model:')
        self.par = SimpleNamespace()
        self.val = SimpleNamespace()
        self.sim = SimpleNamespace()

        # if do_print: print('calling .setup()')
        self.setup()

    def setup(self):
        """ baseline parameters """

        val = self.val
        par = self.par
        sim = self.sim

        # model parameters for analytical solution
        par.k = sm.symbols('k')
        par.alpha = sm.symbols('alpha')
        par.delta = sm.symbols('delta')
        par.sigma =  sm.symbols('sigma')
        par.s_k = sm.symbols('s_k')
        par.g = sm.symbols('g')
        par.A = sm.symbols('A')
        par.l_i = sm.symbols('l_i')
        par.l_m = sm.symbols('l_m')
        par.K = sm.symbols('K')
        par.Y = sm.symbols('Y')
        par.L = sm.symbols('L')
        par.k_tilde = sm.symbols('k_tilde')
        par.y_tilde = sm.symbols('y_tilde')
        par.k_tilde_ss = sm.symbols('k_tilde_ss')

        # model parameter values for numerical solution
        val.s_k = 0.1
        val.g = 0.05
        val.alpha = 0.33
        val.delta = 0.3
        val.sigma = 0.02
        val.l_i = 2
        val.l_m = 1

        # simulation parameters for further analysis
        par.simT = 100 #number of periods
        sim.K = np.zeros(par.simT)
        sim.L = np.zeros(par.simT)
        sim.A = np.zeros(par.simT)
        sim.Y = np.zeros(par.simT)
        sim.L = val.l_i + val.l_m  # total labor force

    def solve_analytical_ss(self):
        """ function that solves the model analytically and returns k_tilde in steady state """

        par = self.par

        # set up the steady state equation
        k_tilde_ss_eq = sm.Eq(par.k_tilde, (1/(1+par.g)) * (par.s_k * par.k_tilde**par.alpha + (1-par.delta) * par.k_tilde))

        # solve the equation for k_tilde
        k_tilde_ss = sm.solve(k_tilde_ss_eq, par.k_tilde)[0]

        # pretty print the solution
        sm.pprint(k_tilde_ss)

        return k_tilde_ss
    
    def solve_numerical_ss(self):
        """ function that solves the model numerically and returns k_tilde in steady state """

        par = self.val

        # define the steady state equation
        def steady_state_eq(k_tilde):
            return k_tilde - (1/(1+par.g)) * (par.s_k * k_tilde**par.alpha + (1-par.delta) * k_tilde)

        # make an initial guess for the solution
        initial_guess = 0.5

        # solve the equation numerically
        k_tilde_ss = optimize.root(steady_state_eq, initial_guess).x[0]

        sm.pprint(k_tilde_ss)

        return k_tilde_ss
