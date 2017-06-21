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

def pca(X, numPrincipalComponents, showScree = False, saveScree = False):
    """ 
    Performs Principal Component Analysis for data dimensionality reduction.
    Assumes that rows pertain to data points and columns to variables.
    """
    #Do some sanity checking.
    if X.shape[0] == 0:
        raise ValueError("Cannot perform PCA on an empty matrix.")
        
    if X.shape[1] < numPrincipalComponents:
        raise ValueError("Cannot reduce %s dimensional data to %d dimensions." % (X.shape[1], numPrincipalComponents))
        
    if X.shape[1] == numPrincipalComponents:
        return X
    
    #Subtract the mean from each column.
    means = np.array([np.mean(X, axis = 0),]*X.shape[0])
    X_meanReduced = np.subtract(X, means)
    
    #Get the covariance matrix of the mean subtracted data.
    cov = np.cov(X_meanReduced, rowvar = False)
    
    #Get the eigendecomposition of the covariance matrix and sort.
    eigVals, eigVecs = np.linalg.eig(cov)
    sortedIndices = eigVals.argsort()[::-1]
    
    #Reduce dimensionality.
    X_reduced = np.dot(X, eigVecs[:, sortedIndices[0:numPrincipalComponents]])
    
    #Plot, if requested.
    if showScree:
        __plotScree(eigVals, sortedIndices[::-1], numPrincipalComponents, saveScree)
    
    return X_reduced
    
def __plotScree(eigVals, sortedIndices, numPrincipalComponents, savePlot = False):
    """
    Displays a scree plot(sorted and normalised Eigenvalues).
    Optionally, one can save the plot to a file named 'scree.png'
    """
    #Sort and sum eigenvalues.
    eigVals = np.sort(eigVals)
    eigSum = np.sum(eigVals)
    
    #Plot.
    x = np.array(range(1, eigVals.shape[0] + 1))
    plt.figure()
    plt.plot(x, eigVals[sortedIndices])
    plt.xticks(x)
    plt.xlabel("Sorted Eigenvalue IDs")
    plt.ylabel("Normalised Eigenvalues")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    if savePlot:
        plt.savefig("scree.png")
    #plt.show()
    
