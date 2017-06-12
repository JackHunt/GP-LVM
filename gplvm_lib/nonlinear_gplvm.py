from gplvm import *
from pca import pca

class NonlinearGPLVM(GPLVM):
    """
    Class representing a nonlinear Gaussian Process Latent Variable Model.
    Defaults to using a Radial Basis Function kernel.
    """

    __kernel = RadialBasisFunction()
    __params = {'gamma' : 1.0, 'theta' : 1.0}
    
    def __init__(self, Y):
        """
        NoninearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)

    def setKernelAndInitialParams(self, kernel, params):
        """
        Sanity checks and updates the kernel and initial hyperparameters for this GP-LVM.
        """
        if len(list(params.keys())) != len(kernel.hyperparameters):
            raise ValueError("Number of hyperparameters provided must match that required by the kernel.")
        
        for var in list(params.keys()):
            if var not in kernel.hyperparameters:
                raise ValueError("Hyperparameters must match those of the given kernel.")
        self.__kernel = kernel;
        self.__params = params;
        
    def compute(self, reducedDimensionality):
        """
        Method to compute latent spaces with a nonlinear GP-LVM.
        """
        #Do sanity checking in base class.
        super().compute(reducedDimensionality)

        #Compute Y*Yt if not already cached and cache it.
        self._computeYYt()
        
        #Initialise with PCA.
        tmpLatent = pca(self._Y, reducedDimensionality)
        
        #Compute covariance matrix of PCA reduced data and it's inverse.
        K = np.array([self.__kernel.f(x, y, self.__params) for x in tmpLatent for y in tmpLatent])
        K = K.reshape((tmpLatent.shape[0], tmpLatent.shape[0]))
#        K_inv = np.linalg.inv(K)

        #Compute Y*Y^t if not already computed, else use cached version.
        self._computeYYt()
