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

import numpy as np
import matplotlib.pyplot as plt

def pca(X, num_principal_components, show_scree = False, save_scree = False):
    """ 
    Performs Principal Component Analysis for data dimensionality reduction.
    Assumes that rows pertain to data points and columns to variables.
    """
    #Do some sanity checking.
    if X.shape[0] == 0:
        raise ValueError("Cannot perform PCA on an empty matrix.")
        
    if X.shape[1] < num_principal_components:
        raise ValueError("Cannot reduce %s dimensional data to %d dimensions." % (X.shape[1], num_principal_components))
        
    if X.shape[1] == num_principal_components:
        return X
    
    #Subtract the mean from each column.
    means = np.array([np.mean(X, axis = 0),]*X.shape[0])
    X_mean_reduced = np.subtract(X, means)
    
    #Get the covariance matrix of the mean subtracted data.
    cov = np.cov(X_mean_reduced, rowvar = False)
    
    #Get the eigendecomposition of the covariance matrix and sort.
    eig_vals, eig_vecs = np.linalg.eig(cov)
    sorted_indices = eig_vals.argsort()[::-1]
    
    #Reduce dimensionality.
    X_reduced = np.dot(X, eig_vecs[:, sorted_indices[0:num_principal_components]])
    
    #Plot, if requested.
    if show_scree:
        __plot_scree(eig_vals, sorted_indices[::-1], num_principal_components, save_scree)
    
    return X_reduced
    
def __plot_scree(eig_vals, sorted_indices, num_principal_components, save_plot = False):
    """
    Displays a scree plot(sorted and normalised Eigenvalues).
    Optionally, one can save the plot to a file named 'scree.png'
    """
    #Sort and sum eigenvalues.
    eig_vals = np.sort(eig_vals)
    eig_sum = np.sum(eig_vals)
    
    #Plot.
    x = np.array(range(1, eig_vals.shape[0] + 1))
    plt.figure()
    plt.plot(x, eig_vals[sorted_indices])
    plt.xticks(x)
    plt.xlabel("Sorted Eigenvalue IDs")
    plt.ylabel("Normalised Eigenvalues")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    if save_plot:
        plt.savefig("scree.png")
    #plt.show()
