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

class LinearGPLVM(GPLVM):
    """Class representing a linear Gaussian Process Latent Variable Model.
    """
    def __init__(self, Y: np.array):
        """Construct a Linear GPLVM (Probabilistic PCA)

        Args:
            Y (np.array): Input Data.
        """
        super().__init__(Y)

        self.YYt_eigenvalues = None
        self.YYt_eigenvectors = None
        self._sorted_indicesYYt = []

    def _compute_eigendecomposition_YYt(self):
        if self.YYt_eigenvalues is None and self.YYt_eigenvectors is None:
            self.YYt_eigenvalues, self.YYt_eigenvectors = np.linalg.eig(self.YYt)
            self._sorted_indicesYYt = self.YYt_eigenvalues.argsort()[::-1]

    def compute(self,
                reduced_dimensionality: int,
                beta: float):
        """Method to compute a latent space embedding with a linear GP-LVM.

        Args:
            reduced_dimensionality (int): Target dimensionality of the latebt space.
            beta (float): Regularizer.
        """        
        # Do sanity checking in base class.
        super().compute(reduced_dimensionality)

        # Data dimensionality.
        D = self.Y.shape[1]

        # Compute Y*Y^t if not already computed, else use cached version.
        self._computeYYt()

        # Compute eigendecomposition of Y*Y^t and sort.
        self._compute_eigendecomposition_YYt()

        # Compute eigendecomposition of Y*Y^t and sort.
        eig_valsDYYt, eig_vecsDYYt = np.linalg.eig((1.0 / D) * self.YYt)
        sorted_indicesDYYt = eig_valsDYYt.argsort()[::-1]

        # Construct L matrix.
        l_vec = eig_valsDYYt[sorted_indicesDYYt[0:reduced_dimensionality]]
        l_vec -= 1.0 / beta
        l_vec = 1.0 / np.sqrt(l_vec)
        L = np.diag(l_vec)

        # Arbitrary rotation matrix.
        V = np.eye(reduced_dimensionality) * 5

        # Finally, compute latent space representation - X = U*L*V^t.
        self.X = np.dot(
            self.YYt_eigenvectors[:, self._sorted_indicesYYt[0:reduced_dimensionality]],
            np.dot(L, V.transpose()))

    @property
    def YYt_eigenvalues(self):
        return self._eigenvalues_YYt

    @YYt_eigenvalues.setter
    def YYt_eigenvalues(self, val):
        if not val is None and not isinstance(val, np.ndarray):
            raise ValueError("YYt_eigenvalues must be a numpy array.")

        if not val is None and self.YYt_eigenvalues and val.shape != self.YYt_eigenvalues.shape:
            raise ValueError(
                "YYt_eigenvalues cannot change shape when being reassigned.")

        self._eigenvalues_YYt = val

    @property
    def YYt_eigenvectors(self):
        return self._eigenvectors_YYt

    @YYt_eigenvectors.setter
    def YYt_eigenvectors(self, val):
        if not val is None and not isinstance(val, np.ndarray):
            raise ValueError("YYt_eigenvectors must be a numpy array.")

        if not val is None and self.YYt_eigenvectors and val.shape != self.YYt_eigenvectors.shape:
            raise ValueError(
                "YYt_eigenvectors cannot change shape when being reassigned.")

        self._eigenvectors_YYt = val
