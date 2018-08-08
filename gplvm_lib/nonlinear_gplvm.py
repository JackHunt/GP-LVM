'''
BSD 3-Clause License

Copyright (c) 2017, Jack Miles Hunt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from gplvm import *
from pca import pca
from sys import float_info

class NonlinearGPLVM(GPLVM):
    """
    Class representing a nonlinear Gaussian Process Latent Variable Model.
    Defaults to using a Radial Basis Function kernel.
    """

    __kernel_initial = RadialBasisFunction()
    __params_initial = {'theta1' : 2.0, 'theta2' : 2.0, 'theta3' : 2.0, 'theta4' : 2.0}
    __kernel = __kernel_initial
    __params = __params_initial
    __init_latent = np.array([])
    __prev_latent_grad = np.array([])
    __prev_hyp_grad = {}

    def __init__(self, Y):
        """
        NoninearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)

    def set_kernel_and_initial_params(self, kernel, params):
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

    def compute(self, reduced_dimensionality, batch_size, jitter = 1, max_iterations = 150, min_step = 1e-6, 
        learn_rate = 0.001, momentum = 0.001, verbose = True, do_latent = True, do_hyper = True):
        """
        Method to compute latent spaces with a nonlinear GP-LVM.
        Training is performed using Stochatic Gradient Descent.
        """
        #Do sanity checking in base class.
        super().compute(reduced_dimensionality)

        #Compute Y*Yt if not already cached and cache it.
        self._computeYYt()
        
        #Initialise with PCA if not already and make copy of initial latent space, also reset momentum.
        tmp_latent = self.__initialise_latent(reduced_dimensionality)
        self.__reset_momentum()
        
        #For storing initial energy.
        initial_energy = 0.0
        E = 0.0

        #Optimise for latent space and hyperparameters.
        for iter in range(0, max_iterations):
            #Compute covariance matrix of PCA reduced data and it's pseudoinverse.
            K = np.array([self.__kernel.f(a, b, self.__params) for a in self._X for b in self._X])
            K = K.reshape((self._X.shape[0], self._X.shape[0]))
            K += np.eye(self._X.shape[0]) * jitter
            K_inv = np.linalg.pinv(K)
            #K_inv = np.linalg.solve(K, np.eye(K.shape[0]))

            #Compute Y*Y^t if not already computed, else use cached version.
            self._computeYYt()

            #Compute energy.
            E = self.__energy(K, K_inv)
            if iter == 0:
                initial_energy = E

            #Train a mini batch with SGD.
            step_norms = self.__update_mini_batch(K_inv, self._X.shape[0], reduced_dimensionality, learn_rate, momentum, batch_size, do_latent, do_hyper)
            do_latent = False if not do_latent or step_norms['latentStep'] <= min_step else True
            do_hyper = False if not do_hyper or step_norms['hyperStep'] <= min_step else True

            #Progress report and early out if converged.
            if verbose:
                print("Iteration: %s \nLog Likelihood: %s \nLatent step L2: %s \nHyperparameter step L2: %s" % (iter, E, step_norms['latentStep'], step_norms['hyperStep']))
                print("--------------------------------------------------------------------------------")
            if not do_latent and not do_hyper:
                print("Converged!")
                break
        print("Initial Log Likelihood: %s \nPost-Optimisation Log Likelihood: %s" % (initial_energy, E))
        print("Final Hyperparameters:\n %s" % self.__params)

    def reset(self):
        """
        Resets the NonlinearGPLVM to it's initial state.
        Kernel, params, latent variables and momentum are reset.
        """
        self.__kernel = self.__kernel_initial
        self.__params = self.__params_initial
        self.__X = np.copy(self.__init_latent)
        self.__resetMomentum()

    def __initialise_latent(self, reduced_dimensionality):
        """
        Initialises latent variables with Principal Component Analysis and keeps a copy for resetting purposes.
        """
        if self.__init_latent.shape[0] == 0 or self.__init_latent.shape[1] != reduced_dimensionality:
            self.__init_latent = pca(self._Y, reduced_dimensionality)
        self._X = np.copy(self.__init_latent)

    def __reset_momentum(self):
        """
        Resets the momentum term(previous gradients) for SGD to zero.
        """
        self.__prev_latent_grad = np.zeros_like(self._X)
        self.__prev_hyp_grad.clear()

    def __update_mini_batch(self, K_inv, N, Q, learn_rate, momentum, batch_size, do_latent, do_hyper):
        """
        Performs SGD(Stochastic Gradient Descent) on a randomised minibatch subset of the latent variables.
        """
        #Compute partial derivatives.
        grads = self.__energy_deriv(K_inv)

        #Generate random mini batch row id's.
        batch_ids = np.random.randint(self._X.shape[0], size = min(self._X.shape[0], batch_size)) if batch_size != self._X.shape[0] else range(self._X.shape[0])

        #Process mini batch.
        latent_sum_sq = hyper_sum_sq = 0.0
        for id in batch_ids:
            #Update latent variables.
            if do_latent:
                dLdb = np.array([grad['b'] for grad in grads['dK']]).reshape(N, N, Q)
                latent_sum_sq += self.__update_latent_variables(dLdb, grads['dLdK'], learn_rate, momentum, id)

            #Update hyperparameters.
            if do_hyper:
                hyper_sum_sq += self.__update_hyperparameters(grads['dLdK'], grads['dK'], learn_rate, momentum, id)

        return {'latentStep' : np.sqrt(latent_sum_sq), 'hyperStep' : np.sqrt(hyper_sum_sq)}

    def __energy(self, K, K_inv):
        """
        Computes the Log Likelihood function. Eq 6 in paper.
        """

        D = self._Y.shape[1]
        N = self._Y.shape[0]

        t1 = -D * N * np.log(2.0 * np.pi)
        t2 = -D / 2.0 * np.linalg.slogdet(K)[1]
        t3 = -0.5 * np.trace(K_inv * self._YYt)

        return (t1 + t2 + t3)

    def __energy_deriv(self, K_inv):
        """
        Computes the partial derivatives dL/dK and dK/dp for p, some hyperparameter.
        """
        D = self._Y.shape[1]
        
        #Compute dL/dK - Reverse KL Divergence diff w.r.t kernel.
        dLdK = np.subtract(np.dot(K_inv, np.dot(self._YYt, K_inv)), D * K_inv)
    
        #Compute kernel partial derivatives.
        dK = [self.__kernel.df(a, b, self.__params) for a in self._X for b in self._X]        

        return {'dLdK' : dLdK, 'dK' : dK}

    def __update_latent_variables(self, dLdb, dLdK, learn_rate, momentum, id):
        """
        Perform gradient updates over latent variables.
        """
        step = np.dot(dLdK[id, :], dLdb[id, :, :])
        self._X[id, :] -= (learn_rate * step + momentum * self.__prev_latent_grad[id, :])
        self.__prev_latent_grad[id, :] = np.copy(step)
        return np.sum(step**2)

    def __update_hyperparameters(self, dLdK, dK, learn_rate, momentum, id):
        """
        Perform gradient updates over hyperparameters.
        """
        step_sum = 0.0
        for var in self.__kernel.hyperparameters:
            if var not in self.__prev_hyp_grad.keys():
                self.__prev_hyp_grad[var] = 0.0

            dKdV = np.array([g[var] for g in dK]).reshape(dLdK.shape[0], dLdK.shape[0])
            tmp = np.dot(dLdK, dKdV)
            #step = 0.5 * (2.0 * tmp.sum(axis=1) - tmp.diagonal())
            step = tmp[id][id]
            self.__params[var] -= (learn_rate * step + momentum * self.__prev_hyp_grad[var])
            self.__prev_hyp_grad[var] = step
            step_sum += step**2
        return step_sum