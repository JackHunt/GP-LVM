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

from abc import ABC, abstractmethod
import numpy as np

class GPLVM(ABC):
    """
    Abstract base class for a Gaussian Process Latent Variable Model.
    Derived classes are implmented as per the following paper:
    'Probabilistic Non-linear Principal Component Analysis with Gaussian
    Process Latent Variable Models'

    Lawrence 2005
    """
    def __init__(self, Y: np.array):
        """Base class constructor.
        Takes a data matrix and assumes that rows pertain to data
        points and columns to variables.

        Args:
            Y (np.array): Input data.

        Raises:
            ValueError: If `Y` is empty.
        """
        # Original data.
        self.Y = Y

        # Y*Y^t - cached Y*Yt to reduce repeated computation.
        self.YYt = None

        # Latent space representation.
        self.X = None


    def _computeYYt(self):
        """Computes YY^t if not already computed. Skips if already cached.

        Raises:
            ValueError: If YY^t is None or is misshapen.
        """        
        if self.YYt is None:
            self.YYt = np.dot(self.Y, self.Y.transpose())

        if self.YYt.shape[0] != self.Y.shape[0] or \
            self.YYt.shape[1] != self.Y.shape[0]:
            raise ValueError(
                "Mismatch between data matrix Y and YY^t. "
                "Have you changed data matrix externally?")

    def get_latent_space_representation(self) -> np.array:
        """Returns the most recently computed latent space representation of the data.

        Returns:
            np.array: Latent space embedding.
        """        
        return self.X

    @abstractmethod
    def compute(self, reduced_dimensionality: int):
        """Abstract method to compute latent spaces with a GP-LVM.

        Args:
            reduced_dimensionality (int): Target dimensionality.

        Raises:
            ValueError: If the target dimensionality is greater than, or equal to
            that of the input data.
        """
        if reduced_dimensionality >= self.Y.shape[1]:
            raise ValueError(
                f"Cannot reduce {self.Y.shape[1]} dimensional data to "
                f"{reduced_dimensionality} dimensions.")

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, val):
        if not val is None and not isinstance(val, np.ndarray):
            raise ValueError("Y must be a numpy array.")

        if not val is None and not val.shape[0]:
            raise ValueError(
                "Cannot compute a GP-LVM on an empty data matrix.")

        self._Y = val

    @property
    def YYt(self):
        return self._YYt

    @YYt.setter
    def YYt(self, val):
        if not val is None and not isinstance(val, np.ndarray):
            raise ValueError("YYt must be a numpy array.")

        if not val is None and self.YYt and val.shape != self.YYt.shape:
            raise ValueError(
                "YYt cannot change shape when being reassigned.")

        self._YYt = val

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, val):
        if not val is None and not isinstance(val, np.ndarray):
            raise ValueError("X must be a numpy array.")

        if not val is None and self.X and val.shape[0] != self.X.shape[0]:
            raise ValueError(
                "X cannot change it's leading dimension when being reassigned.")

        self._X = val
