
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

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
        H = HM**(1-par.alpha)*HF**par.alpha
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM,HF)
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

    def solve(self,do_print=False):
        """ solve model continously """

        pass    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass



# Question 1

model = HouseholdSpecializationModelClass()

#a. Create list with values of alpha and sigma
alpha_list = [0.25, 0.5, 0.75]
sigma_list = [0.5,1.0,1.5]

df = pd.DataFrame(columns = pd.Index(alpha_list, name="sigma/alpha"), index = pd.Index(sigma_list, name=""))


for i in alpha_list:
    for j in sigma_list:
        model.par.alpha = i
        model.par.sigma = j
        results = model.solve_discrete()
        ratio = results.HF / results.HM
        df.loc[j,i] = f"{ratio:.2f}"

print(df)
#print(tabulate(df, headers = alpha_list, tablefmt = "fancy_grid")) 

# Vi kan også overveje at tilføje et heatmap over de forskellige variationer, der 
# kan vise hvilke variatioer med højest værdi





# Question 2 Ny

model = HouseholdSpecializationModelClass()

alpha_list = [0.25, 0.5, 0.75]
sigma_list = [0.5, 1.0, 1.5]
wF_list = [0.8, 0.9, 1.0, 1.1, 1.2]

results_list = []

for alpha in alpha_list:
    for sigma in sigma_list:
        for wF in wF_list:
            results = solve_and_plot(model, alpha, sigma, wF)
            results_list.append((alpha, sigma, *results))

# Plot the results
markers = ['o', '^', 's', 'd', 'x']
styles = ['-', '--', '-.', ':']
for i, alpha in enumerate(alpha_list):
    for j, sigma in enumerate(sigma_list):
        mask = [(r[0]==alpha) and (r[1]==sigma) for r in results_list]
        wF_arr, ratio_arr = zip(*[r[2:] for r in np.array(results_list)[mask]])
        plt.plot(np.log(np.array(wF_arr)/model.par.wM), ratio_arr, marker=markers[j], linestyle=styles[i], label=f'$\\alpha={alpha}$, $\\sigma={sigma}$')

plt.xlabel('$\\log\\frac{w_F}{w_M}$')
plt.ylabel('$\\log\\frac{H_F}{H_M}$')
plt.title('Household Specialization Model')

# Create a separate legend outside of the plot area
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()



#Question 2 OLD version # virkede ikke
#import matplotlib.pyplot as plt

#model2 = HouseholdSpecializationModelClass()

#log_H_ratio = []
#log_w_ratio = []

#for wF in model2.par.wF.vec:
 #   wF = model2.par.wF
  #  wM = model2.par.wM
   # optimum = model2.solve.discrete()
    #log_HFM = np.log(optimum.HF / optimum.HM)
    #log_H_ratio = np.append(log_H_ratio, log_HFM)
    #log_wF = np.log(wF / wM)
    #log_w_ratio = np.append(log_w_ratio , log_wF)





#model = HouseholdSpecializationModelClass()

#HF_HM_ratios = model.solve.wF_vec(discrete=True)

#model.run_regression

#plt.plot(np.log(model.par,wF_vec),np.log(HF_HM_ratios),marker = "x" , linestyle = "--")


#for i, txt in enumerate (model.par.wF_vec):
 #   plt.annotate(txt, (np.log(model.par.wF_vec[i]), np.log(HF_HM_ratios[i])))

#plt.grid(True)
#plt.show()







