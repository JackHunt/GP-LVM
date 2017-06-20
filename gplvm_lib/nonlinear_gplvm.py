from gplvm import *
from pca import pca

class NonlinearGPLVM(GPLVM):
    """
    Class representing a nonlinear Gaussian Process Latent Variable Model.
    Defaults to using a Radial Basis Function kernel.
    """

    __kernel = RadialBasisFunction()
    __params = {'gamma' : 1.0, 'theta' : 1.0}
    __initLatent = np.array([])
    
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

    def compute(self, reducedDimensionality, maxIterations = 150, minStep = 1e-5, learnRate = 0.001, verbose = True):
        """
        Method to compute latent spaces with a nonlinear GP-LVM.
        """
        #Do sanity checking in base class.
        super().compute(reducedDimensionality)

        #Compute Y*Yt if not already cached and cache it.
        self._computeYYt()
        
        #Initialise with PCA if not already and make copy of initial latent space.
        tmpLatent = self.__initialiseLatent(reducedDimensionality)
        self._X = np.copy(self.__initLatent)
        
        #Optimise for latent space.
        for iter in range(0, maxIterations):
            #Compute covariance matrix of PCA reduced data and it's inverse.
            K = np.array([self.__kernel.f(a, b, self.__params) for a in self.__initLatent for b in self.__initLatent])
            K = K.reshape((self.__initLatent.shape[0], self.__initLatent.shape[0]))
            K_inv = np.linalg.inv(K)

            #Compute Y*Y^t if not already computed, else use cached version.
            self._computeYYt()

            #Compute energy.
            #E = self.__energy(K)
            E = 0

            #Compute gradients.
            nabla = self.__energyDeriv(K_inv)

            #Update latent space.
            self._X = np.subtract(self._X, learnRate * np.array([dict['a'] for dict in nabla]))
            
            #Update hyperparameters - TO:DO Make more pythonic.
            hypGrad = {}
            for grad in nabla:
                for var in grad.keys():
                    if var in self.__kernel.hyperparameters:
                        hypGrad[var] += grad[var]

            #Progress report and early out if converged.
            if verbose:
                print("Iteration: %s Reversed KL: %s Step L2: %s" % (iter, E, stepNorm))
            if stepNorm < minStep:
                break

    def __initialiseLatent(self, reducedDimensionality):
        if self.__initLatent.shape[0] == 0 or self.__initLatent.shape[1] != reducedDimensionality:
            self.__initLatent = pca(self._Y, reducedDimensionality)

    def __energy(self, K):
        D = self._Y.shape[1]
        N = self._Y.shape[0]
        S = D * self._YYt
    
        #Terms of reversed KL Divergence.
        t1 = 0.5 * np.log(np.linalg.det(S))
        t2 = -0.5 * np.log(np.linalg.det(K))
        t3 = 0.5 * np.trace(np.dot(K, np.linalg.inv(S)))
        t4 = -N / 2.0

        return t1 + t2 + t3 + t4

    def __energyDeriv(self, K_inv):
        D = self._Y.shape[1]
        
        #Compute dL/dK - Reverse KL Divergence diff w.r.t kernel.
        dLdK = np.subtract(np.dot(K_inv, np.dot(self._YYt, K_inv)), D * K_inv)
        dLdK = dLdK.reshape(K_inv.shape[0] * K_inv.shape[1])
    
        #Compute kernel partial derivatives.
        dK = [self.__kernel.df(a, b, self.__params) for a in self.__initLatent for b in self.__initLatent]

        #Apply chain rule - reuse dK.
        for d, dict in zip(dLdK, dK):
            dict.update((key, val * d) for key, val in dict.items())
        
        return dK