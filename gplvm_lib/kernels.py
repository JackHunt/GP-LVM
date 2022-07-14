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

class Kernel(ABC):
    """Abstract base class for covariance kernels.
    """
    def __init__(self, hyperparameters: dict[str, float]):
        """Base Kernel constructor. Provides limited sanity checking.

        Args:
            hyperparameters (dict[str, float]): A dictionary of Kernel Hyperparameters.

        Raises:
            ValueError: If `hyperparameters` is `None`.
        """        
        if not hyperparameters:
            raise ValueError(
                "Must specify hyperparameter dictionary for child kernel class.")

        self._hyperparameters = hyperparameters

    def __check_valid(self,
                      a: np.array,
                      b: np.array):
        """Checks that two vectors provided to a kernel are a valid combination.
        Sanity checks shapes.

        Args:
            a (np.array): An n-dimensional vector.
            b (np.array): An n-dimensional vector.

        Raises:
            ValueError: If vector dimensions are mismatched.
        """
        if a.shape[0] == 0 or b.shape[0] == 0 or a.shape[0] != b.shape[0]:
            raise ValueError(
                "Kernel input vector dimension mismatch. "
                f"Vector a has shape {a.shape} and Vector b "
                f"has shape {b.shape}")

    @abstractmethod
    def f(self,
          a: np.array,
          b: np.array):
        """Abstract method to evaluate a covariance matrix entry for two
        given vectors and hyperparameter set.

        Args:
            a (np.array): An n-dimensional vector.
            b (np.array): An n-dimensional vector.
        """
        self.__check_valid(a, b)

    @abstractmethod
    def df(self,
           a: np.array,
           b: np.array):
        """Abstract method to evaluate a covariance matrix Jacobian entry for
        two given vectors and hyperparameter set.

        Args:
            a (np.array): An n-dimensional vector.
            b (np.array): An n-dimensional vector.
        """
        self.__check_valid(a, b)

    def _hyperparameter_check(self, hyp_name: str):
        """Verifies that a given hyperparemeter belongs to the kernel instance.

        Args:
            hyp_name (str): Name of the hyperparameter.

        Raises:
            ValueError: If the hyperparameter is unknown to the Kernel.
        """        
        if not hyp_name in self.hyperparameters:
            raise ValueError(
                f"{hyp_name} is not a valid hyperparameter of this kernel.")

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, val):
        if len(val) != len(self.hyperparameters):
            raise ValueError(
                "Inconsistent hyperparameter count. This kernel has "
                f"{len(self.hyperparameters)} hyperparameters.")

        for k in val:
            self._hyperparameter_check(k)

        self._hyperparameters = val

    def set_hyperparameter(self,
                           hyp_name: str,
                           val: float):
        """Sets the value of a hyperparameter.

        Args:
            hyp_name (str): The name of the hyperparameter to update.
            val (float): The updated value of the hyperparameter.
        """        
        self._hyperparameter_check(hyp_name)
        self.hyperparameters[hyp_name] = val

class RadialBasisFunction(Kernel):
    """Radial Basis Function kernel.
    """
    def __init__(self,
                 theta_1:float = 2.0,
                 theta_2:float = 2.0,
                 theta_3:float = 2.0,
                 theta_4:float = 2.0,
                 delta_distance:float = 1e-5):
        """Constructs a new Radial Basis Function.

        Args:
            theta_1 (float, optional): Hyperparameter. Defaults to 2.0.
            theta_2 (float, optional): Hyperparameter. Defaults to 2.0.
            theta_3 (float, optional): Hyperparameter. Defaults to 2.0.
            theta_4 (float, optional): Hyperparameter. Defaults to 2.0.
            delta_distance (float, optional): Delta function tolerance. Defaults to 1e-5.
        """        
        self.delta_distance = delta_distance

        super().__init__({
            'theta_1': theta_1,
            'theta_2': theta_2,
            'theta_3': theta_3,
            'theta_4': theta_4
        })

    def _delta(self, dist: float) -> float:
        """Delta Function.

        Args:
            dist (float): Input value.

        Returns:
            float: Output Value of 1.0 or 0.0.
        """        
        if dist < self.delta_distance:
            return 1.0
        return 0.0

    def f(self,
          a: np.array,
          b: np.array) -> float:
        """Evaluates the Radial Basis Function for some combination of
        vectors and set of hyperparameters.

        Args:
            a (np.array): An n-dimensional vector.
            b (np.array): An n-dimensional vector.

        Returns:
            float: _description_
        """        
        super().f(a, b)
        diff = a - b
        dist = np.dot(diff.transpose(), diff)
        t_1 = self.theta_1 * np.exp((-self.theta_2 / 2.0) * dist)
        t_2 = self.theta_3 + self.theta_4 * self._delta(dist)
        return t_1 + t_2

    def df(self,
           a: np.array,
           b: np.array) -> dict[str, float]:
        """Computes partial derivatives of the RBF.

        Args:
            a (np.array): An n-dimensional vector.
            b (np.array): An n-dimensional vector.

        Returns:
            dict[str, float]: Gradient value.
        """        
        super().df(a, b)

        diff = a - b
        dist = np.dot(diff.transpose(), diff)

        df_ds_1 = np.exp(-0.5 * self.theta_2 * dist)
        df_ds_2 = self.theta_1 * dist * df_ds_1
        df_ds_3 = 1.0
        df_ds_4 = self._delta(dist)

        df_db = -1.0 * self.theta_1 * self.theta_2 * diff * df_ds_1

        return {
            'b' : df_db,
            'theta_1' : df_ds_1,
            'theta_2' : df_ds_2,
            'theta_3' : df_ds_3,
            'theta_4' : df_ds_4
        }

    @property
    def delta_distance(self):
        return self._delta_distance

    @delta_distance.setter
    def delta_distance(self, val):
        if val <= 0.0:
            raise ValueError("delta_distance must be nonzero and nonnegative.")

        self._delta_distance = val

    @property
    def theta_1(self):
        return self.hyperparameters['theta_1']

    @property
    def theta_2(self):
        return self.hyperparameters['theta_2']

    @property
    def theta_3(self):
        return self.hyperparameters['theta_3']

    @property
    def theta_4(self):
        return self.hyperparameters['theta_4']
