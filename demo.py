#!/usr/bin/python3
import sys
sys.path.insert(0, './gplvm_lib')
import urllib.request
import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt
import gplvm_lib as gp

irisURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
irisFname = 'iris.data'

#Control plotting here.
showPlots = True
savePlots = True

def getIris():
    """
    Loads the four dimensional Fisher Iris dataset.
    If the 'iris.data' file is not present in the working directory, 
    this function attempts to download it.
    The last column of the dataset(the text labels) are ommitted.
    """
    iris = []
    if not os.path.isfile(irisFname):
        print("Attempting to download the iris dataset.")
        try:
            urllib.request.urlretrieve(irisURL, irisFname)
        except urllib.request.URLError:
            sys.exit("Unable to download iris dataset. Quitting.")
    
    with open(irisFname, newline='') as file:
        reader = csv.reader(file, delimiter = ',')
        for line in reader:
            if len(line) != 0:
                iris.append(list(map(float, line[0:4])))
    return np.asarray(iris)

def plot(data, dimensionality, title, method):
    """
    Helper function to reduce code duplication.
    """
    if dimensionality == 1:
        gp.plot1D(data, title, method, savePlots)
    elif dimensionality == 2:
        gp.plot2D(data, title, method, savePlots)
    elif dimensionality == 3:
        gp.plot3D(data, title, method, savePlots)
    else:
        return None
    
def runPCA(data, reducedDimensions, showScree):
    """
    Runs standard PCA on the given dataset, optionally showing the associated 
    Scree plot(normalised Eigenvalues)
    """
    print("-->Running PCA.")
    latent = gp.pca(data, reducedDimensions, showScree, savePlots)
    plot(latent, reducedDimensions, "Iris Dataset", "PCA")
    
def runLinearGPLVM(data, reducedDimensions, beta):
    """
    Runs the Linear Gaussian Process Latent Variable Model on the given dataset. 
    The resultant data plotted if the latent space is 1, 2 or 3 dimensional.
    """
    print("-->Running Linear GP-LVM.")
    gplvm = gp.LinearGPLVM(data)
    gplvm.compute(reducedDimensions, beta)
    latent = gplvm.getLatentSpaceRepresentation()
    plot(latent, reducedDimensions, "Iris Dataset", "Linear GP-LVM")
    
def runNonlinearGPLVM():
    """
    Runs the Nonlinear Gaussian Process Latent Variable Model on the given dataset, 
    for a given covariance matrix generating kernel.
    The resultant data plotted if the latent space is 1, 2 or 3 dimensional.
    """
    print("-->Running Nonlinear GPLVM.")
    
if __name__ == "__main__":
    """
    Parameters of the algorithms may be tweaked here.
    """
    
    newDimensionality = 2 #Dimension to reduce to.
    beta = 2.0 #Beta parameter for Linear GP-LVM.
    scree = True #Whether to display the Scree plot for PCA.
    data = getIris()
    
    runPCA(data, newDimensionality, scree)
    runLinearGPLVM(data, newDimensionality, beta)
    runNonlinearGPLVM()
    
    if showPlots:
        plt.show()
