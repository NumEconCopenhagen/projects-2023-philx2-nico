from types import SimpleNamespace
import sympy as sm

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