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

#Defining class
class SolowModelClass(): 
              """ Creating the model """

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

            #model parameters for analytical solution
            par.k = sm.symbols('k')
            par.alpha = sm.symbols('alpha')
            par.delta = sm.symbols('delta')
            par.sigma =  sm.symbols('sigma')
            par.s = sm.symbols('s')
            par.g = sm.symbols('g')
            par.n = sm.symbols('n')
            par.d = sm.symbols('D')
            par.dT = sm.symbols('dT') #change in temperature
            par.kss = sm.symbols(r'$\tilde k_t$')