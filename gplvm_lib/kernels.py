from abc import ABC, abstractmethod

class Kernel(ABC):
    """
    Abstract base class for covariance kernels.
    """
    
    #In derived classes this will be a list of hyperparameters.
    hyperparameters = []

    @abstractmethod
    def f(self, a, b, params):
        """
        Abstract method to evaluate a covariance matrix entry for two given vectors and hyperparameter set.
        """
        pass

    @abstractmethod
    def df(self, a, b, params, var):
        """
        Abstract method to evaluate a covariance matrix derivative entry for two given vectors and hyperparameter set.
        """
        pass