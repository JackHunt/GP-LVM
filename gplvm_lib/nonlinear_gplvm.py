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
        self.__kernel = kernel
        self.__params = params

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

        #Initialise steps.
        latentStep = 1e5
        hyperStep = 1e5
        
        #Optimise for latent space.
        for iter in range(0, maxIterations):
            #Compute covariance matrix of PCA reduced data and it's inverse.
            K = np.array([self.__kernel.f(a, b, self.__params) for a in self._X for b in self._X])
            K = K.reshape((self._X.shape[0], self._X.shape[0]))
            K_inv = np.linalg.inv(K)

            #Compute Y*Y^t if not already computed, else use cached version.
            self._computeYYt()

            #Compute energy.
            #E = self.__energy(K)
            E = 0

            #Compute partial derivatives.
            grads = self.__energyDeriv(K_inv)

            #Update latent variables.
            if latentStep > minStep:
                latentStep = self.__updateLatentVariables(grads, K.shape[0], reducedDimensionality, learnRate)

            #Update hyperparameters.
            if hyperStep > minStep:
                hyperStep = self.__updateHyperparameters(grads, K.shape[0], learnRate)

            #Progress report and early out if converged.
            if verbose:
                print("Iteration: %s Reversed KL: %s Latent step L2: %s Hyper step L2: %s" % (iter, E, latentStep, hyperStep))

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
    
        #Compute kernel partial derivatives.
        dK = [self.__kernel.df(a, b, self.__params) for a in self.__initLatent for b in self.__initLatent]        

        return (dLdK, dK)

    def __updateLatentVariables(self, grads, N, Q, learnRate):
        #Compute gradients w.r.t latent variables and update.
        dLda = np.array([grad['a'] for grad in grads[1]]).reshape(N, N, Q)
        for i in range(0, Q):
            dL = grads[0] * dLda[:, :, i]
            self._X[:, i] = self._X[:, i] - 0.5 * learnRate * (2 * dL.sum(axis=1) - dL.diagonal())
        return 1.0

    def __updateHyperparameters(self, grads, N, learnRate):
        #Compute gradients w.r.t hyperparameters and update.
        for hyp in self.__kernel.hyperparameters:
            dL = grads[0] * np.array([d[hyp] for d in grads[1]]).reshape(N, N)
            self.__params[hyp] = self.__params[hyp] - learnRate * dL.trace()
        return 1.0