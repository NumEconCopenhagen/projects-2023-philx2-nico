from types import SimpleNamespace

import numpy as np
from scipy import optimize
from tabulate import tabulate

import pandas as pd 
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

import types

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """
        
        # a. create namespaces
        par = self.par = SimpleNamespace()

        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0
        
        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan


    def calc_utility(self,LM,HM,LF,HF):
       """ calculate utility """
    
       par = self.par
       sol = self.sol
    
       # a. consumption of market goods
       C = par.wM*LM + par.wF*LF
    
       # b. home production
       if par.sigma == 1.0:
           H = HM**(1-par.alpha)*HF**par.alpha
       elif par.sigma == 0:
           H = np.minimum(HM, HF)
       else: 
           with np.errstate(all='ignore'):
               H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
    
       # c. total consumption utility
       Q = C**par.omega*H**(1-par.omega)
       utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
    
       # d. disutlity of work
       epsilon_ = 1+1/par.epsilon
       TM = LM+HM
       TF = LF+HF
       disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
       
       return utility - disutility


    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self, do_print=False):
        """ solve model continuously """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # Define the objective function to be maximized
        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)
    
        # Define the constraints
        def constraint(x):
            LM, HM, LF, HF = x
            return np.array([24 - (LM + HM), 24 - (LF + HF)])
    
        # Set the initial guess
        x0 = np.array([12, 12, 12, 12])
        
        # Use the minimize function to maximize the utility subject to the constraints
        res = optimize.minimize(objective, x0, method='trust-constr', constraints={'type': 'ineq', 'fun': constraint})
        
        # Get the maximizing argument
        LM, HM, LF, HF = res.x
        
        # Save the solution
        sol.LM = LM
        sol.HM = HM
        sol.LF = LF
        sol.HF = HF
        
        # Print the solution
        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol


    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # Calculate log ratios for each wF
        log_ratios = []
        for wF in par.wF_vec:
            par.wF = wF
            results = self.solve(discrete)
            log_ratios.append(np.log(results.HF / results.HM))

        # Fit a linear regression
        X = np.log(np.array(par.wF_vec) / par.wM).reshape(-1, 1)
        y = np.array(log_ratios)
        lin_reg = LinearRegression().fit(X, y)

        # Save the estimated coefficients
        sol.beta0 = lin_reg.intercept_
        sol.beta1 = lin_reg.coef_[0]

    def minimize_squared_differences(model, alpha_sigma, discrete=False):
        """ minimize the squared differences between the estimated and target coefficients """

        par = model.par
        sol = model.sol

        # Set the new alpha and sigma values
        par.alpha = alpha_sigma[0]
        par.sigma = alpha_sigma[1]

        # Solve the model for the given alpha and sigma values
        model.solve_wF_vec(discrete)

        # Calculate the squared differences
        squared_diff = (sol.beta0 - par.beta0_target)**2 + (sol.beta1 - par.beta1_target)**2

        return squared_diff


    def run_regression(self):
        # Define alpha, sigma, and wF values
        alpha_list = [0.25, 0.5, 0.75]
        sigma_list = [0.5, 1.0, 1.5]
        wF_list = [0.8, 0.9, 1.0, 1.1, 1.2]

        # Create an empty dictionary to store results
        results_dict = {}

        # Loop over all combinations of alpha, sigma, and wF
        for alpha in alpha_list:
            for sigma in sigma_list:
                for wF in wF_list:
                    # Set parameter values
                    self.par.alpha = alpha
                    self.par.sigma = sigma
                    self.par.wF = wF

                    # Solve the model
                    sol = self.solve()

                    # Calculate log ratios
                    log_HF_HM = np.log(sol.HF / sol.HM)
                    log_wF_wM = np.log(self.par.wF / self.par.wM)

                    # Store results in dictionary
                    if (alpha, sigma) not in results_dict:
                        results_dict[(alpha, sigma)] = []
                    results_dict[(alpha, sigma)].append((wF, log_HF_HM, log_wF_wM))

        # Initialize table
        table = []


        # Perform regression for each combination of alpha and sigma
        for alpha in alpha_list:
            for sigma in sigma_list:
                # Initialize arrays for regression
                X = np.empty((0,))
                Y = np.empty((0,))

                # Fill arrays with data
                for wF, log_HF_HM, log_wF_wM in results_dict[(alpha, sigma)]:
                    X = np.append(X, log_wF_wM)
                    Y = np.append(Y, log_HF_HM)

                # Perform linear regression
                A = np.vstack([np.ones(X.size), X]).T
                beta, sse, _, _ = np.linalg.lstsq(A, Y, rcond=None)

                # Add regression results to table
                row = [alpha, sigma, beta[0], beta[1], sse]
                table.append(row)

                df1 = pd.DataFrame(table, columns=["Alpha", "Sigma", "Beta0", "Beta1", "SSE"])

                # Format floating-point numbers in DataFrame
                df1 = df1.round({"Alpha": 2, "Sigma": 1, "Beta0": 4, "Beta1": 4, "SSE": 4})

        # Format and return table
        return df1


    def sum_of_squared_diffs(self, params, beta0_target, beta1_target):
        alpha, sigma = params
        self.par.alpha = alpha
        self.par.sigma = sigma
        df1 = self.run_regression()

        beta0_hat = df1.loc[df1["Alpha"] == alpha, "Beta0"].values[0]
        beta1_hat = df1.loc[df1["Alpha"] == alpha, "Beta1"].values[0]

        return (beta0_target - beta0_hat) ** 2 + (beta1_target - beta1_hat) ** 2

    def estimate(self, alpha=None, sigma=None):
        # Set target coefficients
        beta0_target = 0.4
        beta1_target = -0.1

        # Optimize the function
        initial_guess = [0.5, 1.0]
        bounds = [(0.4, 0.6), (0.5, 1.5)]  # Assuming a narrow interval for alpha and sigma
        result = minimize(self.sum_of_squared_diffs, initial_guess, args=(beta0_target, beta1_target), bounds=bounds)

        # Print best-fit alpha and sigma values
        best_alpha, best_sigma = result.x
        print(f"Best-fit alpha: {best_alpha:.4f}, Best-fit sigma: {best_sigma:.4f}")

        return best_alpha, best_sigma
    
#    def __init__(self):
        # Set up the parameters
        self.par = types.SimpleNamespace(alpha=0.5, gamma=1.0, wF_vec=np.linspace(0.5, 2.0, 50), wM=1.0)
        self.sol = {}

 #   def solve_wF_vec(self):
        alpha = self.par.alpha
        gamma = self.par.gamma
        wF_vec = self.par.wF_vec
        wM = self.par.wM
        
        # Calculate the household production
        self.sol['HF_vec'] = wF_vec**alpha * wM**(1-alpha)
        self.sol['HM_vec'] = wF_vec**(1-alpha) * wM**alpha
        
        # Calculate the specialization
        self.sol['specialization'] = self.sol['HF_vec']**gamma + self.sol['HM_vec']**gamma