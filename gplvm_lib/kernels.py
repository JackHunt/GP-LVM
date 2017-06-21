from abc import ABC, abstractmethod
import numpy as np

class Kernel(ABC):
    """
    Abstract base class for covariance kernels.
    """
    
    #In derived classes this will be a list of hyperparameters.
    hyperparameters = []

    def __init__(self, hyperParams):
        """
        Base Kernel constructor. Provides limited sanity checking.
        """
        if len(hyperParams) == 0:
            raise ValueError("Must specify hyperparameters for child kernel class.")
        self.hyperparameters = hyperParams

    def __checkValid(self, a, b, params):
        """
        Checks that two vectors provided to a kernel are a valid combination.
        Sanity checks shapes.
        """
        if a.shape[0] == 0 or b.shape[0] == 0 or a.shape[0] != b.shape[0]:
            raise ValueError("Kernel input vector dimension error. Check input vectors to kernel.")

        for var in list(params.keys()):
            if var not in self.hyperparameters:
                raise ValueError("Kernel does not contain hyperparameter '%s'" % var)

    @abstractmethod
    def f(self, a, b, params):
        """
        Abstract method to evaluate a covariance matrix entry for two given vectors and hyperparameter set.
        """
        self.__checkValid(a, b, params)

    @abstractmethod
    def df(self, a, b, params):
        """
        Abstract method to evaluate a covariance matrix derivative entry for two given vectors and hyperparameter set.
        """
        self.__checkValid(a, b, params)

class RadialBasisFunction(Kernel):
    """
    Radial Basis Function kernel.
    """

    def __init__(self):
        """
        Constructs a new Radial Basis Function.
        Calls super class constructor for sanity checking.
        """
        super().__init__(['gamma', 'theta'])

    def f(self, a, b, params):
        """
        Evaluates the Radial Basis Function for some combination of vectors and set of hyperparameters.
        Calls on base class for sanity checking.
        """
        super().f(a, b, params)
        diff = a - b
        return params['theta'] * np.exp((-params['gamma'] / 2.0) * np.dot(diff.transpose(), diff))

    def df(self, a, b, params):
        """
        Computes partial derivatives of RBF.
        Again, calls super class for sanity checking.
        """
        super().df(a, b, params)
        dist = np.dot((a - b).transpose(), (a - b))
        dFdS = np.exp(-0.5 * params['gamma'] * dist)
        dFdG = params['theta'] * dist * np.exp(-0.5 * params['gamma'] * dist)
        dFdB = -1.0 * params['theta'] * params['gamma'] * (a - b) * np.exp(-0.5 * params['gamma'] * dist)

        return {'b' : dFdB, 'theta' : dFdS, 'gamma' : dFdG}
