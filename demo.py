#!/usr/bin/python3
import sys
sys.path.insert(0, './gplvm_lib')
import urllib.request
import os.path
import numpy as np
import gplvm_lib as gp

irisURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
irisFname = 'iris.data'

def getIris():
    iris = []
    if not os.path.isfile(irisFname):
        try:
            urllib.request.urlretrieve(irisURL, irisFname)
        except urllib.request.URLError:
            sys.exit("Unable to download iris dataset. Quitting.")
    return iris

if __name__ == "__main__":
    data = getIris()
    gplvm = gp.LinearGPLVM(a + np.eye(3)*5)
    gplvm.compute(2, 2.0)
    #gp.pca(np.array([[1, 2, 3], [1, 2, 3]]), 2, True)
