from types import SimpleNamespace
import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class LaborEconomicsModelClass:
    def __init__(self):
        """ create the model """
        self.par = SimpleNamespace()

        # Define symbols for the model parameters
        self.par.alpha = sm.symbols('alpha')
        self.par.kappa = sm.symbols('kappa')
        self.par.nu = sm.symbols('nu')
        self.par.tau = sm.symbols('tau')
        self.par.w = sm.symbols('w')
        self.par.L = sm.symbols('L')
        self.par.G = sm.symbols('G')
        self.par.w_tilde = sm.symbols('wtilde')

    def solve_model_symbolically(self, G_value):
        """ function that solves the model symbolically """
        # Define the utility function
        utility = sm.log((self.par.kappa + self.par.w_tilde*self.par.L)**self.par.alpha*self.par.G**(1-self.par.alpha)) - self.par.nu*(self.par.L**2)/2
        
        # Substitute G with its value
        utility_subs_G = utility.subs(self.par.G, G_value)
        
        # Take derivative of utility with respect to L
        FOC = sm.diff(utility_subs_G, self.par.L)
        
        # Solve for optimal L
        L_star = sm.solve(FOC, self.par.L)[0]
        
        # Return the solution for L as a symbolic expression
        return L_star

    def print_FOC(self, G_value):
        """ function that prints the first order condition """
        # Define the utility function
        utility = sm.log((self.par.kappa + self.par.w_tilde*self.par.L)**self.par.alpha*self.par.G**(1-self.par.alpha)) - self.par.nu*(self.par.L**2)/2
        
        # Substitute G with its value
        utility_subs_G = utility.subs(self.par.G, G_value)
        
        # Take derivative of utility with respect to L
        FOC = sm.diff(utility_subs_G, self.par.L)
        
        # Print the FOC in symbolic form
        print("FOC: ", FOC)

class NewLaborEconomicsModelClass:
    def __init__(self, sigma=1.001, rho=1.001, epsilon=1.0):
        """ create the model """
        self.par = SimpleNamespace()

        # Define symbols for the model parameters
        self.par.alpha = sm.symbols('alpha')
        self.par.kappa = sm.symbols('kappa')
        self.par.nu = sm.symbols('nu')
        self.par.tau = sm.symbols('tau')
        self.par.w = sm.symbols('w')
        self.par.L = sm.symbols('L')
        self.par.G = sm.symbols('G')
        self.par.w_tilde = sm.symbols('wtilde')
        self.par.sigma = sm.symbols('sigma')
        self.par.rho = sm.symbols('rho')
        self.par.epsilon = sm.symbols('epsilon')

        # Assign the parameters
        self.par.sigma = sigma
        self.par.rho = rho
        self.par.epsilon = epsilon

    def solve_model_symbolically(self, G_value):
        """ function that solves the model symbolically """
        # Define the utility function
        utility = ((self.par.alpha*(self.par.kappa + self.par.w_tilde*self.par.L)**((self.par.sigma - 1)/self.par.sigma) + 
                (1 - self.par.alpha)*self.par.G**((self.par.sigma - 1)/self.par.sigma))**(self.par.sigma/(self.par.sigma - 1))**(1 - self.par.rho) - 1)/(1 - self.par.rho) - \
                self.par.nu*(self.par.L**(1 + self.par.epsilon))/(1 + self.par.epsilon)    
        
        # Substitute G with its value
        utility_subs_G = utility.subs(self.par.G, G_value)
        
        # Take derivative of utility with respect to L
        FOC = sm.diff(utility_subs_G, self.par.L)
        
        # Solve for optimal L
        L_star = sm.solve(FOC, self.par.L)[0]
        
        # Return the solution for L as a symbolic expression
        return L_star

    def print_FOC(self, G_value):
        """ function that prints the first order condition """
        # Define the utility function
        utility = sm.log((self.par.kappa + self.par.w_tilde*self.par.L)**self.par.alpha*self.par.G**(1-self.par.alpha)) - self.par.nu*(self.par.L**2)/2
        
        # Substitute G with its value
        utility_subs_G = utility.subs(self.par.G, G_value)
        
        # Take derivative of utility with respect to L
        FOC = sm.diff(utility_subs_G, self.par.L)
        
        # Print the FOC in symbolic form
        print("FOC: ", FOC)

class HairSalonOptimizer:
    def __init__(self, rho, iota, sigma_epsilon, R, eta, w, K, T):
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.eta = eta
        self.w = w
        self.K = K
        self.T = T
        np.random.seed(123)
        self.epsilon_values = [np.random.normal(loc=-0.5*self.sigma_epsilon**2, scale=self.sigma_epsilon, size=self.T) for _ in range(self.K)]
    
    def calculate_l_t(self, kappa_t):
        return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)
    
    def calculate_H(self):
        total_value = 0
        for k in range(self.K):
            epsilon = self.epsilon_values[k]
            kappa = np.empty(self.T)
            kappa[0] = np.exp(self.rho*np.log(1) + epsilon[0])
            l_last = 0
            l = []
            for t in range(self.T):
                if t > 0:
                    kappa[t] = np.exp(self.rho*np.log(kappa[t-1]) + epsilon[t])
                l_star = ((1 - self.eta) * kappa[t] / self.w) ** (1 / self.eta)
                l.append(l_star if abs(l_last - l_star) > 0 else l_last)
                l_last = l[-1]
            h = np.sum([(kappa[t]*l[t]**(1-self.eta) - self.w*l[t] - self.iota*(l[t] != l[t-1])) * self.R**(-t) for t in range(self.T)])
            total_value += h
        return total_value / self.K
    
    def find_optimal_delta(self, delta_values):
        H_values = []
        for Delta in delta_values:
            H_values.append(self.calculate_H_with_policy(Delta))
        max_H_index = np.argmax(H_values)
        max_H = H_values[max_H_index]
        max_Delta = delta_values[max_H_index]
        return max_Delta, max_H, H_values
    
    def calculate_H_with_policy(self, Delta):
        total_value = 0
        for k in range(self.K):
            epsilon = self.epsilon_values[k]
            kappa = np.empty(self.T)
            kappa[0] = np.exp(self.rho*np.log(1) + epsilon[0])
            l_last = 0
            l = []
            for t in range(self.T):
                if t > 0:
                    kappa[t] = np.exp(self.rho*np.log(kappa[t-1]) + epsilon[t])
                l_star = ((1 - self.eta) * kappa[t] / self.w) ** (1 / self.eta)
                if abs(l_last - l_star) > Delta:
                    l.append(l_star)
                else:
                    l.append(l_last)
                l_last = l[-1]
            h = np.sum([(kappa[t]*l[t]**(1-self.eta) - self.w*l[t] - self.iota*(l[t] != l[t-1])) * self.R**(-t) for t in range(self.T)])
            total_value += h
        return total_value / self.K
    
    def plot_H_vs_delta(self, delta_values, H_values, max_Delta, max_H):
        plt.plot(delta_values, H_values)
        plt.plot(max_Delta, max_H, 'ro')  # mark the maximum point
        plt.xlabel('Delta')
        plt.ylabel('H')
        plt.title('H vs Delta')
        plt.grid(True)
        plt.show()


class Griewank:
    def __init__(self):
        pass

    @staticmethod
    def evaluate(x):
        A = x[0]**2/4000 + x[1]**2/4000
        B = np.cos(x[0]/np.sqrt(1))*np.cos(x[1]/np.sqrt(2))
        return A-B+1

class RefinedGlobalOptimizer:
    def __init__(self, objective_function, bounds, tolerance, warm_up_iters, max_iters):
        self.objective_function = objective_function
        self.bounds = bounds
        self.tolerance = tolerance
        self.warm_up_iters = warm_up_iters
        self.max_iters = max_iters
        self.x_k0_values = []  # for storing initial guesses

    def set_warm_up_iters(self, warm_up_iters):
        self.warm_up_iters = warm_up_iters
    
    def optimize(self):
        self.x_k0_values.clear()  # clear the list at the start of each call
        x_star = np.array([0, 0])
        for k in range(self.max_iters):   # 3.A: Draw random x^k uniformly within chosen bounds
            x_k = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            if k >= self.warm_up_iters:         # 3.B, 3.C and 3.D
                chi_k = 0.50 * 2/(1 + np.exp((k-self.warm_up_iters)/100))
                x_k0 = chi_k * x_k + (1 - chi_k) * x_star
            else:
                x_k0 = x_k
            self.x_k0_values.append(x_k0)
            result = minimize(self.objective_function.evaluate, x_k0, method='BFGS', tol=self.tolerance) #3.E: optimizer
            x_k_star = result.x
            if k == 0 or self.objective_function.evaluate(x_k_star) < self.objective_function.evaluate(x_star): #3.F
                x_star = x_k_star
            if self.objective_function.evaluate(x_star) < self.tolerance: #3.G
                break
        return x_star
    

