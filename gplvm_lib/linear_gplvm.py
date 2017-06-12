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
        
        
        
