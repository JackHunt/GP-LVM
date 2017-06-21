from gplvm import *
from pca import pca

class NonlinearGPLVM(GPLVM):
    """
    Class representing a nonlinear Gaussian Process Latent Variable Model.
    Defaults to using a Radial Basis Function kernel.
    """

    __kernel_initial = RadialBasisFunction()
    __params_initial = {'gamma' : 2.0, 'theta' : 2.0}
    __kernel = __kernel_initial
    __params = __params_initial
    __initLatent = np.array([])
    __prevLatentGrad = np.array([])

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

    def compute(self, reducedDimensionality, batchSize, maxIterations = 150, minStep = 1e-6, learnRate = 0.001, momentum = 0.001, verbose = True):
        """
        Method to compute latent spaces with a nonlinear GP-LVM.
        Training is performed using Stochatic Gradient Descent.
        """
        #Do sanity checking in base class.
        super().compute(reducedDimensionality)

        #Compute Y*Yt if not already cached and cache it.
        self._computeYYt()
        
        #Initialise with PCA if not already and make copy of initial latent space, also reset momentum.
        tmpLatent = self.__initialiseLatent(reducedDimensionality)
        self.__resetMomentum()
        
        #Optimise for latent space and hyperparameters.
        doLatent = doHyper = True
        for iter in range(0, maxIterations):
            #Compute covariance matrix of PCA reduced data and it's pseudoinverse.
            K = np.array([self.__kernel.f(a, b, self.__params) for a in self._X for b in self._X])
            K = K.reshape((self._X.shape[0], self._X.shape[0]))
            K += np.eye(self._X.shape[0])*5
            K_inv = np.linalg.pinv(K)

            #Compute Y*Y^t if not already computed, else use cached version.
            self._computeYYt()

            #Compute energy.
            E = self.__energy(K, K_inv)

            #Train a mini batch with SGD.
            stepNorms = self.__updateMiniBatch(K_inv, self._X.shape[0], reducedDimensionality, learnRate, momentum, batchSize, doLatent, doHyper)
            doLatent = False if not doLatent or stepNorms['latentStep'] <= minStep else True
            doHyper = False if not doHyper or stepNorms['hyperStep'] <= minStep else True

            #Progress report and early out if converged.
            if verbose:
                print("Iteration: %s \nLog Likelihood: %s \nLatent step L2: %s \nHyperparameter step L2: %s" % (iter, E, stepNorms['latentStep'], stepNorms['hyperStep']))
                print("--------------------------------------------------------------------------------")
            if not doLatent and not doHyper:
                print("Converged!")
                break

    def reset(self):
        """
        Resets the NonlinearGPLVM to it's initial state.
        Kernel, params, latent variables and momentum are reset.
        """
        self.__kernel = self.__kernel_initial
        self.__params = self.__params_initial
        self.__X = np.copy(self.__initLatent)
        self.__resetMomentum()

    def __initialiseLatent(self, reducedDimensionality):
        """
        Initialises latent variables with Principal Component Analysis and keeps a copy for resetting purposes.
        """
        if self.__initLatent.shape[0] == 0 or self.__initLatent.shape[1] != reducedDimensionality:
            self.__initLatent = pca(self._Y, reducedDimensionality)
        self._X = np.copy(self.__initLatent)

    def __resetMomentum(self):
        """
        Resets the momentum term(previous gradients) for SGD to zero.
        """
        self.__prevLatentGrad = np.zeros_like(self._X)

    def __updateMiniBatch(self, K_inv, N, Q, learnRate, momentum, batchSize, doLatent, doHyper):
        """
        Performs SGD(Stochastic Gradient Descent) on a randomised minibatch subset of the latent variables.
        """
        #Compute partial derivatives.
        grads = self.__energyDeriv(K_inv)
        if doLatent:
            dLda = np.array([grad['b'] for grad in grads['dK']]).reshape(N, N, Q)

        #Generate random mini batch row id's.
        batchIDs = np.random.randint(self._X.shape[0], size = min(self._X.shape[0], batchSize))

        #Process mini batch.
        latentSumSq = hyperSumSq = 0.0
        for id in batchIDs:
            #Update latent variables.
            if doLatent:
                latentSumSq += self.__updateLatentVariables(dLda, grads['dLdK'], learnRate, momentum, id)

            #Update hyperparameters.
            if doHyper:
                hyperSumSq += self.__updateHyperparameters(grads['dLdK'], grads['dK'], learnRate, id)

        return {'latentStep' : np.sqrt(latentSumSq), 'hyperStep' : np.sqrt(hyperSumSq)}

    def __energy(self, K, K_inv):
        """
        Computes the Log Likelihood function. Eq 6 in paper.
        """
        D = self._Y.shape[1]
        N = self._Y.shape[0]

        t1 = -D * N * np.log(2.0 * np.pi)
        t2 = -D / 2.0 * np.log(np.linalg.det(K))
        t3 = -0.5 * np.trace(K_inv * self._YYt)

        return t1 + t2 + t3

    def __energyDeriv(self, K_inv):
        """
        Computes the partial derivatives dL/dK and dK/dp for p, some hyperparameter.
        """
        D = self._Y.shape[1]
        
        #Compute dL/dK - Reverse KL Divergence diff w.r.t kernel.
        dLdK = np.subtract(np.dot(K_inv, np.dot(self._YYt, K_inv)), D * K_inv)
    
        #Compute kernel partial derivatives.
        dK = [self.__kernel.df(a, b, self.__params) for a in self._X for b in self._X]        

        return {'dLdK' : dLdK, 'dK' : dK}

    def __updateLatentVariables(self, dLda, dLdK, learnRate, momentum, id):
        """
        Perform gradient updates over latent variables.
        """
        step = np.dot(dLdK, dLda[id, :, :])
        self._X += - (learnRate * step + momentum * self.__prevLatentGrad)
        self.__prevLatentGrad = np.copy(step)
        return np.sum(step**2)

    def __updateHyperparameters(self, dLdK, dK, learnRate, id):
        """
        Perform gradient updates over hyperparameters.
        """
        return 0.0