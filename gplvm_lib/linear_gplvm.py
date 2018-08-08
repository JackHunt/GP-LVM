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

class LinearGPLVM(GPLVM):
    """
    Class representing a linear Gaussian Process Latent Variable Model.
    """

    _eig_valsYYt = np.array([])
    _eig_vecsYYt = np.array([])
    _sorted_indicesYYt = []
    
    def __init__(self, Y):
        """
        LinearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)

    def _compute_eigendecompositionYYt(self):
        if self._eig_valsYYt.shape[0] == 0 and self._eig_vecsYYt.shape[0] == 0:
            self._eig_valsYYt, self._eig_vecsYYt = np.linalg.eig(self._YYt)
            self._sorted_indicesYYt = self._eig_valsYYt.argsort()[::-1]
        
    def compute(self, reduced_dimensionality, beta):
        """
        Method to compute latent spaces with a linear GP-LVM.
        """
        #Do sanity checking in base class.
        super().compute(reduced_dimensionality)
        
        #Data dimensionality.
        D = self._Y.shape[1]
        
        #Compute Y*Y^t if not already computed, else use cached version.
        self._computeYYt()
        
        #Compute eigendecomposition of Y*Y^t and sort.
        self._compute_eigendecompositionYYt()
        
        #Compute eigendecomposition of Y*Y^t and sort.
        eig_valsDYYt, eig_vecsDYYt = np.linalg.eig((1.0 / D) * self._YYt)
        sorted_indicesDYYt = eig_valsDYYt.argsort()[::-1]
        
        #Construct L matrix.
        lVec = eig_valsDYYt[sorted_indicesDYYt[0:reduced_dimensionality]]
        lVec -= 1.0 / beta
        lVec = 1.0 / np.sqrt(lVec)
        L = np.diag(lVec)
        
        #Arbitrary rotation matrix.
        V = np.eye(reduced_dimensionality) * 5
        
        #Finally, compute latent space representation - X = U*L*V^t.
        self._X = np.dot(self._eig_vecsYYt[:, self._sorted_indicesYYt[0:reduced_dimensionality]], np.dot(L, V.transpose()))
