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

from abc import ABC, abstractmethod
import numpy as np
from kernels import *

class GPLVM(ABC):
    """
    Abstract base class for a Gaussian Process Latent Variable Model.
    Derived classes are implmented as per the following paper:
    'Probabilistic Non-linear Principal Component Analysis with Gaussian Process Latent Variable Models'
    Lawrence 2005
    """
    
    #Original data.
    _Y = np.array([])

    #Y*Y^t - cached Y*Yt to reduce repeated computation.
    _YYt = np.array([])
    
    #Latent space representation.
    _X = np.array([])
    
    def __init__(self, Y):
        """
        Base class constructor.
        Takes a data matrix and assumes that rows pertain to data points and columns variables.
        """
        #Sanity check the data and store.
        if Y.shape[0] == 0:
            raise ValueError("Cannot compute a GP-LVM on an empty data matrix.")
        self._Y = Y

    def _computeYYt(self):
        """
        Computes YY^t if not already computed. Skips if already cached.
        """
        if self._YYt.shape[0] == 0:
            self._YYt = np.dot(self._Y, self._Y.transpose())

        if self._YYt.shape[0] != self._Y.shape[0] or self._YYt.shape[0] != self._Y.shape[0]:
            raise ValueError("Mismatch between data matrix Y and YY^t. Have you changed data matrix externally?")
        
    def getLatentSpaceRepresentation(self):
        """
        Returns the most recently computed latent space representation of the data.
        """
        return self._X
    
    @abstractmethod
    def compute(self, reducedDimensionality):
        """
        Abstract method to compute latent spaces with a GP-LVM.
        """
        if reducedDimensionality >= self._Y.shape[1]:
            raise ValueError("Cannot reduce %s dimensional data to %d dimensions." % (self._Y.shape[1], reducedDimensionality))
