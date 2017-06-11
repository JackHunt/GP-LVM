#!/usr/bin/python3
import sys
sys.path.insert(0, './gplvm_lib')
import numpy as np
import gplvm_lib as gp

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    gplvm = gp.LinearGPLVM(a + np.eye(3)*5)
    gplvm.compute(2, 2.0)
    #gp.pca(np.array([[1, 2, 3], [1, 2, 3]]), 2, True)
