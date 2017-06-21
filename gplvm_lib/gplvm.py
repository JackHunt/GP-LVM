from abc import ABC, abstractmethod
import numpy as np
from kernels import *

class GPLVM(ABC):
    """
    Abstract base class for a Gaussian Process Latent Variable Model.
    Derived classes are implmented as per the following paper:
    'Probabilistic Non-linear Principal Component Analysis with Gaussian Process Latent Variable Models'
    Lawrence 2005
    """
    
    #Original data.
    _Y = np.array([])

    #Y*Y^t - cached Y*Yt to reduce repeated computation.
    _YYt = np.array([])
    
    #Latent space representation.
    _X = np.array([])
    
    def __init__(self, Y):
        """
        Base class constructor.
        Takes a data matrix and assumes that rows pertain to data points and columns variables.
        """
        #Sanity check the data and store.
        if Y.shape[0] == 0:
            raise ValueError("Cannot compute a GP-LVM on an empty data matrix.")
        self._Y = Y

    def _computeYYt(self):
        """
        Computes YY^t if not already computed. Skips if already cached.
        """
        if self._YYt.shape[0] == 0:
            self._YYt = np.dot(self._Y, self._Y.transpose())

        if self._YYt.shape[0] != self._Y.shape[0] or self._YYt.shape[0] != self._Y.shape[0]:
            raise ValueError("Mismatch between data matrix Y and YY^t. Have you changed data matrix externally?")
        
    def getLatentSpaceRepresentation(self):
        """
        Returns the most recently computed latent space representation of the data.
        """
        return self._X
    
    @abstractmethod
    def compute(self, reducedDimensionality):
        """
        Abstract method to compute latent spaces with a GP-LVM.
        """
        if reducedDimensionality >= self._Y.shape[1]:
            raise ValueError("Cannot reduce %s dimensional data to %d dimensions." % (self._Y.shape[1], reducedDimensionality))
