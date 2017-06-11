from gplvm import *

class LinearGPLVM(GPLVM):
    """
    Class representing a linear Gaussian Process Latent Variable Model.
    """
    
    def __init__(self, Y):
        """
        LinearGPLVM class constructor.
        See base class documentation.
        """
        super().__init__(Y)
    
    def compute(self, reducedDimensionality, beta):
        """
        Method to compute latent spaces with a linear GP-LVM.
        """
        #Do sanity checking in base class.
        super().compute(reducedDimensionality, beta)
        
        #Data dimensionality.
        D = self._Y.shape[1]
        
        #Compute Y*Y^t - can be resued.
        YYt = np.dot(self._Y, self._Y.transpose())
        
        #Compute eigendecomposition of Y*Y^t and sort.
        eigValsYYt, eigVecsYYt = np.linalg.eig(YYt)
        sortedIndicesYYt = eigValsYYt.argsort()
        
        #Compute eigendecomposition of Y*Y^t and sort.
        eigValsDYYt, eigVecsDYYt = np.linalg.eig((1.0 / D) * YYt)
        sortedIndicesDYYt = eigValsDYYt.argsort()
        
        #Construct L matrix.
        lVec = eigValsDYYt[sortedIndicesDYYt[0:reducedDimensionality]]
        lVec -= 1.0 / beta
        lVec = np.sqrt(lVec)
        L = np.diag(lVec)
        
        #Arbitrary rotation matrix.
        V = np.eye(reducedDimensionality)
        
        #Finally, compute latent space representation - X = U*L*V^t.
        self._X = np.dot(eigVecsYYt[:, sortedIndicesYYt[0:reducedDimensionality]], np.dot(L, V.transpose()))
        
        
        