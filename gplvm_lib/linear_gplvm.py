from gplvm import *

class LinearGPLVM(GPLVM):
    """
    Class representing a linear Gaussian Process Latent Variable Model.
    """
    
    def __init__(self, Y):
        """
        LinearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)
    
    def compute(self, reducedDimensionality):
        """
        Method to compute latent spaces with a linear GP-LVM.
        """
        #Do sanity checking in base class.
        super().compute(reducedDimensionality)
        
        #Data dimensionality.
        D = self._Y.shape[1]
        
        YYt = np.dot(self._Y, self._Y.transpose())
        
        