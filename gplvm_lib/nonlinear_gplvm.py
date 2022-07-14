"""
BSD 3-Clause License

Copyright (c) 2022, Jack Miles Hunt
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
"""

import numpy as np

from gplvm_lib.gplvm import GPLVM
from gplvm_lib.kernels import Kernel
from gplvm_lib.kernels import RadialBasisFunction
from gplvm_lib.pca import pca

class NonlinearGPLVM(GPLVM):
    """Class representing a nonlinear Gaussian Process Latent Variable Model.
    Defaults to using a Radial Basis Function kernel.
    """
    def __init__(self,
                 Y: np.array,
                 kernel:Kernel = RadialBasisFunction()):
        """NonlinearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)

        self.kernel = kernel

        self._init_latent = np.array([])
        self._prev_latent_grad = np.array([])
        self._prev_hyp_grad = {}

    def compute(self,
                reduced_dimensionality: int,
                batch_size: int,
                jitter: int = 1,
                max_iterations: int = 150,
                min_step: float = 1e-6,
                learn_rate: float = 0.001,
                momentum: float = 0.001,
                verbose: bool = True,
                do_latent: bool = True,
                do_hyper: bool = True):
        """Method to compute latent spaces with a nonlinear GP-LVM.
        Training is performed using Stochatic Gradient Descent.
        """
        # Do sanity checking in base class.
        super().compute(reduced_dimensionality)

        # Compute Y*Yt if not already cached and cache it.
        self._computeYYt()

        # Initialise with PCA if not already and make copy of initial latent space,
        # also reset momentum.
        self._initialise_latent(reduced_dimensionality)
        self._reset_momentum()

        # For storing initial energy.
        initial_energy = 0.0
        E = 0.0

        # Optimise for latent space and hyperparameters.
        for i in range(max_iterations):
            # Compute covariance matrix of PCA reduced data and it's pseudoinverse.
            K = np.array([self.kernel.f(a, b) for a in self.X for b in self.X])
            K = K.reshape((self.X.shape[0], self.X.shape[0]))
            K += np.eye(self.X.shape[0]) * jitter
            K_inv = np.linalg.pinv(K)
            #K_inv = np.linalg.solve(K, np.eye(K.shape[0]))

            # Compute Y*Y^t if not already computed, else use cached version.
            self._computeYYt()

            # Compute energy.
            E = self._energy(K, K_inv)
            if i == 0:
                initial_energy = E

            # Train a mini batch with SGD.
            step_norms = self._update_mini_batch(K_inv,
                                                 self.X.shape[0],
                                                 reduced_dimensionality,
                                                 learn_rate,
                                                 momentum,
                                                 batch_size,
                                                 do_latent,
                                                 do_hyper)
            do_latent = False if not do_latent or step_norms['latentStep'] <= min_step else True
            do_hyper = False if not do_hyper or step_norms['hyperStep'] <= min_step else True

            # Progress report and early out if converged.
            if verbose:
                print(
                    f"Iteration: {i} \nLog Likelihood: {E}\n"
                    f"Latent step L2: {step_norms['latentStep']}\n"
                    f"Hyperparameter step L2: {step_norms['hyperStep']}\n"
                    "----------------------------------------------------------------------------")
            if not do_latent and not do_hyper:
                print("Converged!")
                break
        print(
            f"Initial Log Likelihood: {initial_energy}\n"
            f"Post-Optimisation Log Likelihood: {E}\n"
            f"Final Hyperparameters:\n {self.kernel.hyperparameters}")

    def reset(self):
        """Resets the NonlinearGPLVM to it's initial state.
        Kernel, params, latent variables and momentum are reset.
        """
        self.kernel = self.kernel.__class__()
        self.X = np.copy(self._init_latent)
        self._reset_momentum()

    def _initialise_latent(self, reduced_dimensionality):
        """Initialises latent variables with Principal Component Analysis and keeps a copy
        for resetting purposes.
        """
        if self._init_latent.shape[0] == 0 or self._init_latent.shape[1] != reduced_dimensionality:
            self._init_latent = pca(self.Y, reduced_dimensionality)
        self.X = np.copy(self._init_latent)

    def _reset_momentum(self):
        """Resets the momentum term(previous gradients) for SGD to zero.
        """
        self._prev_latent_grad = np.zeros_like(self.X)
        self._prev_hyp_grad.clear()

    def _update_mini_batch(self,
                           K_inv: np.array,
                           N: int,
                           Q: int,
                           learn_rate: float,
                           momentum: float,
                           batch_size: int,
                           do_latent: bool,
                           do_hyper: bool) -> dict[str, np.array]:
        """Performs SGD(Stochastic Gradient Descent) on a randomised minibatch subset of
        the latent variables.
        """
        # Compute partial derivatives.
        grads = self._energy_grad(K_inv)

        # Generate random mini batch row id's.
        batch_ids = np.random.randint(self.X.shape[0], size=min(self.X.shape[0], batch_size)) if batch_size != self.X.shape[0] else range(self.X.shape[0])

        # Process mini batch.
        latent_sum_sq = 0.0
        hyper_sum_sq = 0.0
        for i in batch_ids:
            # Update latent variables.
            if do_latent:
                dl_db = np.array([grad['b'] for grad in grads['dk']]).reshape(N, N, Q)
                latent_sum_sq += self._update_latent_vars(dl_db,
                                                          grads['dl_dk'],
                                                          learn_rate,
                                                          momentum,
                                                          i)

            # Update hyperparameters.
            if do_hyper:
                hyper_sum_sq += self._update_hyperparams(grads['dl_dk'],
                                                         grads['dk'],
                                                         learn_rate,
                                                         momentum,
                                                         i)

        return {
            'latentStep': np.sqrt(latent_sum_sq),
            'hyperStep' : np.sqrt(hyper_sum_sq)
        }

    def _energy(self,
                 K: np.array,
                 K_inv: np.array) -> float:
        """Computes the Log Likelihood function. Eq 6 in paper.
        """

        D = self.Y.shape[1]
        N = self.Y.shape[0]

        t_1 = -D * N * np.log(2.0 * np.pi)
        t_2 = -D / 2.0 * np.linalg.slogdet(K)[1]
        t_3 = -0.5 * np.trace(K_inv * self.YYt)

        return t_1 + t_2 + t_3

    def _energy_grad(self, K_inv: np.array) -> dict[str, float]:
        """Computes the partial derivatives dL/dk and dk/dp for p, some hyperparameter.
        """
        D = self.Y.shape[1]

        # Compute dL/dk - Reverse KL Divergence grad w.r.t kernel.
        dl_dk = np.subtract(np.dot(K_inv, np.dot(self.YYt, K_inv)), D * K_inv)

        # Compute kernel partial derivatives.
        dk = [self.kernel.df(a, b) for a in self.X for b in self.X]

        return {
            'dl_dk': dl_dk,
            'dk' : dk
        }

    def _update_latent_vars(self,
                                  dl_db: np.array,
                                  dl_dk: np.array,
                                  learn_rate: float,
                                  momentum: float,
                                  id: int) -> float:
        """Perform gradient updates over latent variables.
        """
        step = np.dot(dl_dk[id, :], dl_db[id, :, :])
        self.X[id, :] -= (learn_rate * step + momentum * self._prev_latent_grad[id, :])
        self._prev_latent_grad[id, :] = np.copy(step)
        return np.sum(step**2)

    def _update_hyperparams(self,
                                 dl_dk: np.array,
                                 dk: np.array,
                                 learn_rate: float,
                                 momentum: float,
                                 id: int) -> float:
        """Perform gradient updates over hyperparameters.
        """
        step_sum = 0.0
        for var in self.kernel.hyperparameters:
            if var not in self._prev_hyp_grad:
                self._prev_hyp_grad[var] = 0.0

            dk_dv = np.array([g[var] for g in dk]).reshape(dl_dk.shape[0], dl_dk.shape[0])
            tmp = np.dot(dl_dk, dk_dv)
            #step = 0.5 * (2.0 * tmp.sum(axis=1) - tmp.diagonal())
            step = tmp[id][id]
            self.kernel.hyperparameters[var] -= \
                (learn_rate * step + momentum * self._prev_hyp_grad[var])
            self._prev_hyp_grad[var] = step
            step_sum += step**2
        return step_sum

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, val):
        if not val:
            raise ValueError("Kernel must not be None.")

        if not isinstance(val, Kernel):
            raise ValueError("Kernel must be an instance of a kernels.Kernel")

        self._kernel = val
