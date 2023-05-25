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


class government:
    
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
        par.C = sm.symbols('C')
        par.k = sm.symbols('k')
        par.w = sm.symbols('w')
        par.tilde_w = sm.symbols('tw')
        par.tau = sm.symbols('tau')
        par.G = sm.symbols('G')
        par.nu = sm.symbols('nu')
        par.a = sm.symbols('alpha')

        # model parameter values for numerical solution
        val.a = 0.5
        val.k = 1.0
        val.nu = 1/512
        val.w = 1.0
        val.tau = 0.30

        # Wage tilde
        tilde_w = (1 - tau) * w

        def optimal_labor(w_tilde, kappa, alpha, nu):
            return (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * w_tilde**2)) / (2 * w_tilde)
        