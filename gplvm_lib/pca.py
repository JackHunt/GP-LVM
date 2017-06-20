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
    X_reduced = np.dot(X_meanReduced, eigVecs[:, sortedIndices[0:numPrincipalComponents]])
    
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
    x = np.array(range(1, numPrincipalComponents + 1))
    plt.figure()
    plt.plot(x, eigVals[sortedIndices][0:numPrincipalComponents])
    plt.xticks(x)
    plt.xlabel("Sorted Eigenvalue IDs")
    plt.ylabel("Normalised Eigenvalues")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    if savePlot:
        plt.savefig("scree.png")
    #plt.show()
    
