from types import SimpleNamespace
import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


import numpy as np
from scipy.optimize import minimize, root
import sympy as sm
from types import SimpleNamespace

class LaborEconomicsModelClass:
    def __init__(self, alpha=0.5, kappa=1.0, nu=1/(2*16**2), w=1.0):
        # Numeric constants
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu
        self.w = w

        # Symbolic parameters
        self.par = SimpleNamespace()
        self.val = SimpleNamespace()

        # Define symbols for the model parameters
        self.par.alpha, self.par.kappa, self.par.nu, self.par.tau, self.par.w, self.par.L, self.par.G, self.par.w_tilde = sm.symbols('alpha kappa nu tau w L G wtilde')

        # Initialize numeric parameters with default values
        self.val.alpha = self.alpha
        self.val.kappa = self.kappa
        self.val.nu = self.nu
        self.val.tau = 0.3
        self.val.w = self.w

    def utility(self, L, tau, G, sigma, rho, epsilon):
        C = self.kappa + (1 - tau)*self.w*L
        U = ((self.alpha * C**((sigma-1)/sigma) + (1-self.alpha) * G**((sigma-1)/sigma))**(sigma/(sigma-1)))**(1-rho)/(1-rho) - self.nu * L**(1+epsilon) / (1+epsilon)
        return -U

    def utility_new(self, L, G_value, sigma, rho, epsilon):
        """ function that returns the updated utility """
        C = self.kappa + (1 - self.val.tau) * self.w * L
        U = ((self.val.alpha * C**((sigma-1)/sigma) + (1-self.val.alpha) * G_value**((sigma-1)/sigma))**(sigma/(sigma-1)))**(1-rho)/(1-rho) - self.val.nu * L**(1+epsilon) / (1+epsilon)
        return U
    
    def G_condition(self, G, tau, L):
        return G - tau * self.w * L

    def solve_model_numerically(self, parameters_sets):
        results = []
        for i, (sigma, rho, epsilon) in enumerate(parameters_sets, 1):
            max_utility = -np.inf
            optimal_tau = None
            optimal_L = None
            optimal_G = None
            for tau in np.linspace(0, 1, 100):
                G = 1
                result_L = minimize(self.utility, x0=1, args=(tau, G, sigma, rho, epsilon), bounds=[(0, 24)])
                if result_L.success:
                    L_star = result_L.x[0]
                    result_G = root(self.G_condition, x0=1, args=(tau, L_star))
                    if result_G.success:
                        G_star = result_G.x[0]
                        U = -self.utility(L_star, tau, G_star, sigma, rho, epsilon)
                        if U > max_utility:
                            max_utility = U
                            optimal_tau = tau
                            optimal_L = L_star
                            optimal_G = G_star
            results.append({'Set': i, 'Parameters': (sigma, rho, epsilon), 'Optimal tau': optimal_tau, 'Optimal L': optimal_L, 'Optimal G': optimal_G})
        return results

    def utility_function(self, L, G_value):
        utility = sm.log((self.par.kappa + self.par.w_tilde*L)**self.par.alpha*G_value**(1-self.par.alpha)) - self.par.nu*(L**2)/2
        return utility

    def solve_model_symbolically(self, G_value):
        utility = self.utility_function(self.par.L, G_value)
        FOC = sm.diff(utility, self.par.L)
        L_star = sm.solve(FOC, self.par.L)[0]
        return L_star

    def print_FOC(self, G_value):
        utility = self.utility_function(self.par.L, G_value)
        FOC = sm.diff(utility, self.par.L)
        print("FOC: ", FOC)

    def L_star(self, tilde_w):
        return (-self.kappa + np.sqrt(self.kappa**2 + 4 * self.alpha / self.nu * tilde_w**2)) / (2 * tilde_w)
    
    def G(self, tau):
        tilde_w = (1 - tau) * self.w
        L = self.L_star(tilde_w)
        return tau * self.w * L
    
    def V(self, tau):
        tilde_w = (1 - tau) * self.w
        L = self.L_star(tilde_w)
        C = self.kappa + tilde_w * L
        return np.log(C**self.alpha * self.G(tau)**(1-self.alpha)) - self.nu * L**2 / 2
    
    def neg_utility_new(self, L, G_value, sigma, rho, epsilon, tau):
        C = self.kappa + (1 - tau) * self.w * L
        U = ((self.alpha * C**((sigma-1)/sigma) + (1-self.alpha) * G_value**((sigma-1)/sigma))**(sigma/(sigma-1)))**(1-rho)/(1-rho) - self.nu * L**(1+epsilon) / (1+epsilon)
        return -U
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
    

