# Necessary libraries

from types import SimpleNamespace

import numpy as np
from scipy import optimize
from tabulate import tabulate

import pandas as pd 
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

import types

# We define our HouseholdSpecializationModelClass

class HouseholdSpecializationModelClass:
    # 2.1. Initialization and parameter setting
    def __init__(self):
        """ setup model """
        
        # Namespaces
        par = self.par = SimpleNamespace()

        sol = self.sol = SimpleNamespace()

        # Preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # Household production
        par.alpha = 0.5
        par.sigma = 1.0
        
        # Wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # Targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # Child care
        par.gamma = 0.5 
        par.delta = 1.0 

        # Our solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    # To calculate the utility
    def calc_utility(self,LM,HM,LF,HF):
       """ calculate utility """
    
       par = self.par
       sol = self.sol
    
       # Consumption of market goods
       C = par.wM*LM + par.wF*LF
    
       # Home production
       if par.sigma == 1.0:
           H = HM**(1-par.alpha)*HF**par.alpha
       elif par.sigma == 0:
           H = np.minimum(HM, HF)
       else: 
           with np.errstate(all='ignore'):
               H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
    
       # Total consumption utility
       Q = C**par.omega*H**(1-par.omega)
       utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
    
       # Disutility of work
       epsilon_ = 1+1/par.epsilon
       TM = LM+HM
       TF = LF+HF
       disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
       
       return utility - disutility

    # To solve the model
    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # Possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # We have all possible combinations
    
        LM = LM.ravel() 
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # To calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # If the constrain is broknen we set to negative infinity
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # We want to find the maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # To print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self, do_print=False):
        """ solve model continuously """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # Objective function which we will maximize
        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)
    
        # Constraints
        def constraint(x):
            LM, HM, LF, HF = x
            return np.array([24 - (LM + HM), 24 - (LF + HF)])
    
        # Our initial guess
        x0 = np.array([12, 12, 12, 12])
        
        # The minimize function is used to maximize the utility with the constraints
        res = optimize.minimize(objective, x0, method='trust-constr', constraints={'type': 'ineq', 'fun': constraint})
        
        # Maximizing argument
        LM, HM, LF, HF = res.x
        
        # We save our solution
        sol.LM = LM
        sol.HM = HM
        sol.LF = LF
        sol.HF = HF
        
        # Printing solutiong
        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol

    # We estimate the model
    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # To calculate the log ratios for each wF
        log_ratios = []
        for wF in par.wF_vec:
            par.wF = wF
            results = self.solve(discrete)
            log_ratios.append(np.log(results.HF / results.HM))

        # Linear regression
        X = np.log(np.array(par.wF_vec) / par.wM).reshape(-1, 1)
        y = np.array(log_ratios)
        lin_reg = LinearRegression().fit(X, y)

        # Saving our estimated coefficients
        sol.beta0 = lin_reg.intercept_
        sol.beta1 = lin_reg.coef_[0]

    def run_regression(self):
        # Alpha, sigma and wF values
        alpha_list = [0.25, 0.5, 0.75]
        sigma_list = [0.5, 1.0, 1.5]
        wF_list = [0.8, 0.9, 1.0, 1.1, 1.2]

        # Empty dictionary
        results_dict = {}

        # Looping over all combinations
        for alpha in alpha_list:
            for sigma in sigma_list:
                for wF in wF_list:
                    # Parameter values
                    self.par.alpha = alpha
                    self.par.sigma = sigma
                    self.par.wF = wF

                    # To solve model
                    sol = self.solve()

                    # The log ratios
                    log_HF_HM = np.log(sol.HF / sol.HM)
                    log_wF_wM = np.log(self.par.wF / self.par.wM)

                    # Storing results
                    if (alpha, sigma) not in results_dict:
                        results_dict[(alpha, sigma)] = []
                    results_dict[(alpha, sigma)].append((wF, log_HF_HM, log_wF_wM))


        table = []


        # Regression for each combination of alpha and sigma
        for alpha in alpha_list:
            for sigma in sigma_list:
                # Arrays for regression
                X = np.empty((0,))
                Y = np.empty((0,))

                # Filling arrays
                for wF, log_HF_HM, log_wF_wM in results_dict[(alpha, sigma)]:
                    X = np.append(X, log_wF_wM)
                    Y = np.append(Y, log_HF_HM)

                # Linear regression
                A = np.vstack([np.ones(X.size), X]).T
                beta, sse, _, _ = np.linalg.lstsq(A, Y, rcond=None)

                # Results added to table
                row = [alpha, sigma, beta[0], beta[1], sse]
                table.append(row)

                df1 = pd.DataFrame(table, columns=["Alpha", "Sigma", "Beta0", "Beta1", "SSE"])

                # Formatting in datafram
                df1 = df1.round({"Alpha": 2, "Sigma": 1, "Beta0": 4, "Beta1": 4, "SSE": 4})

        #Return table
        return df1


    def estimate(model, alpha=None, sigma=None, discrete=False, beta0_target=0.4, beta1_target=-0.1):

        def minimize_squared_differences(alpha_sigma):
            """ minimize the squared differences between the estimated and target coefficients """

            par = model.par
            sol = model.sol

            # New alpha and sigma values
            par.alpha = alpha_sigma[0]
            par.sigma = alpha_sigma[1]

            # Solve model
            model.solve_wF_vec(discrete)

            # Calculate sqr diffs
            squared_diff = (sol.beta0 - beta0_target) ** 2 + (sol.beta1 - beta1_target) ** 2

            return squared_diff

        # Optimize the function
        initial_guess = [0.5, 1.0]
        result = optimize.minimize(minimize_squared_differences, initial_guess, method='Nelder-Mead')

        # To print bestfit alpha and sigma values
        best_alpha, best_sigma = result.x
        error = result.fun
        print(f"Best-fit alpha: {best_alpha:.4f}, Best-fit sigma: {best_sigma:.4f}")
        print(f"Error (squared differences): {error:.6f}")

        return best_alpha, best_sigma
    
class NewModel:
    def __init__(self):
        """ setup model """
        
        # Namespaces
        par = self.par = SimpleNamespace()

        sol = self.sol = SimpleNamespace()

        # Preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # Household production
        par.alpha = 0.5
        par.sigma = 1.0
        
        # Wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # Targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # Child care
        par.gamma = 0.5 
        par.delta = 1.0 

        # Nubmer of children
        par.N = 5

        # The solutions
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    # Calculating the utility 
    def calc_utility(self, LM, HM, LF, HF, N):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # Consumption of market goods
        C = par.wM * LM + par.wF * LF

        # Home production
        if par.sigma == 1.0:
            H = HM ** (1 - par.alpha) * HF ** par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM, HF)
        else:
            with np.errstate(all='ignore'):
                H = ((1 - par.alpha) * HM ** ((par.sigma - 1) / par.sigma) + par.alpha * HF ** ((par.sigma - 1) / par.sigma)) ** (par.sigma / (par.sigma - 1))

        # Child care
        child_care = N ** par.gamma

        # Total consumption utility
        Q = C ** par.omega * H ** (1 - par.omega) * child_care ** par.delta
        utility = np.fmax(Q, 1e-8) ** (1 - par.rho) / (1 - par.rho)

        # Disutility of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM ** epsilon_ / epsilon_ + TF ** epsilon_ / epsilon_)

        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # Possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) 
    
        LM = LM.ravel() 
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # Calculating utility
        u = self.calc_utility(LM,HM,LF,HF,par.N)
    
        # If constraint broken then minus infinity
        I = (LM+HM > 24) | (LF+HF > 24) 
        u[I] = -np.inf
    
        # Maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # Print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve(self, do_print=False):
        """ solve model continuously """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # Maximize objective function
        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF, par.N)
    
        # Constraints
        def constraint(x):
            LM, HM, LF, HF = x
            return np.array([24 - (LM + HM), 24 - (LF + HF)])
    
        # Our initial guess
        x0 = np.array([12, 12, 12, 12])
        
        # We use minimize function to maximize the utility 
        res = optimize.minimize(objective, x0, method='trust-constr', constraints={'type': 'ineq', 'fun': constraint})
        
        # Maximizing argument
        LM, HM, LF, HF = res.x
        
        # We want to save the solution
        sol.LM = LM
        sol.HM = HM
        sol.LF = LF
        sol.HF = HF
        
        # Print 
        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol

    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # Calculating log ratios for each wF
        log_ratios = []
        for wF in par.wF_vec:
            par.wF = wF
            results = self.solve(discrete)
            log_ratios.append(np.log(results.HF / results.HM))

        # Linear regression
        X = np.log(np.array(par.wF_vec) / par.wM).reshape(-1, 1)
        y = np.array(log_ratios)
        lin_reg = LinearRegression().fit(X, y)

        # Saving our estimated coefficients
        sol.beta0 = lin_reg.intercept_
        sol.beta1 = lin_reg.coef_[0]

    def run_regression(self):
        # Alpha, sigma, and wF values
        alpha_list = [0.25, 0.5, 0.75]
        sigma_list = [0.5, 1.0, 1.5]
        wF_list = [0.8, 0.9, 1.0, 1.1, 1.2]

        # Empty dictionary
        results_dict = {}

        # Looping over all combinations 
        for alpha in alpha_list:
            for sigma in sigma_list:
                for wF in wF_list:
                    # Parameter values
                    self.par.alpha = alpha
                    self.par.sigma = sigma
                    self.par.wF = wF

                    # Solving model
                    sol = self.solve()

                    # Log ratios
                    log_HF_HM = np.log(sol.HF / sol.HM)
                    log_wF_wM = np.log(self.par.wF / self.par.wM)

                    # Storing results in dict
                    if (alpha, sigma) not in results_dict:
                        results_dict[(alpha, sigma)] = []
                    results_dict[(alpha, sigma)].append((wF, log_HF_HM, log_wF_wM))

      
        table = []

        # Regression on each combination of alpha and sigma
        for alpha in alpha_list:
            for sigma in sigma_list:
                # Arrays for the regression
                X = np.empty((0,))
                Y = np.empty((0,))

                # Filling arrays 
                for wF, log_HF_HM, log_wF_wM in results_dict[(alpha, sigma)]:
                    X = np.append(X, log_wF_wM)
                    Y = np.append(Y, log_HF_HM)

                # Linear regression
                A = np.vstack([np.ones(X.size), X]).T
                beta, sse, _, _ = np.linalg.lstsq(A, Y, rcond=None)

                # Putting results into tables
                row = [alpha, sigma, beta[0], beta[1], sse]
                table.append(row)

        df1 = pd.DataFrame(table, columns=["Alpha", "Sigma", "Beta0", "Beta1", "SSE"])

        # Formatting in dataframe
        df1 = df1.round({"Alpha": 2, "Sigma": 1, "Beta0": 4, "Beta1": 4, "SSE": 4})

        return df1


