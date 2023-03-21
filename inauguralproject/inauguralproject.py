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
        par.alpha = [0.25, 0.50, 0.75]
        par.sigma = [0.5, 1.0, 1.50]

        # d. wages
        par.wM = 1.0
        par.wF = 1.0

        
















def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
    
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y