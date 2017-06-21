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

    _eigValsYYt = np.array([])
    _eigVecsYYt = np.array([])
    _sortedIndicesYYt = []
    
    def __init__(self, Y):
        """
        LinearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)

    def _computeEigendecompositionYYt(self):
        if self._eigValsYYt.shape[0] == 0 and self._eigVecsYYt.shape[0] == 0:
            self._eigValsYYt, self._eigVecsYYt = np.linalg.eig(self._YYt)
            self._sortedIndicesYYt = self._eigValsYYt.argsort()[::-1]
        
    def compute(self, reducedDimensionality, beta):
        """
        Method to compute latent spaces with a linear GP-LVM.
        """
        #Do sanity checking in base class.
        super().compute(reducedDimensionality)
        
        #Data dimensionality.
        D = self._Y.shape[1]
        
        #Compute Y*Y^t if not already computed, else use cached version.
        self._computeYYt()
        
        #Compute eigendecomposition of Y*Y^t and sort.
        self._computeEigendecompositionYYt()
        
        #Compute eigendecomposition of Y*Y^t and sort.
        eigValsDYYt, eigVecsDYYt = np.linalg.eig((1.0 / D) * self._YYt)
        sortedIndicesDYYt = eigValsDYYt.argsort()[::-1]
        
        #Construct L matrix.
        lVec = eigValsDYYt[sortedIndicesDYYt[0:reducedDimensionality]]
        lVec -= 1.0 / beta
        lVec = 1.0 / np.sqrt(lVec)
        L = np.diag(lVec)
        
        #Arbitrary rotation matrix.
        V = np.eye(reducedDimensionality)*5
        
        #Finally, compute latent space representation - X = U*L*V^t.
        self._X = np.dot(self._eigVecsYYt[:, self._sortedIndicesYYt[0:reducedDimensionality]], np.dot(L, V.transpose()))
        
        
        
