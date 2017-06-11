from gplvm import *

class NonlinearGPLVM(GPLVM):
    """
    Class representing a nonlinear Gaussian Process Latent Variable Model.
    """

    def __init__(self, Y):
        """
        NoninearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)
    
    def compute(self):
        """
        Method to compute latent spaces with a linear GP-LVM.
        """
        pass