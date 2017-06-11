#!/usr/bin/python3
import sys
sys.path.insert(0, './gplvm_lib')
import numpy as np
import gplvm_lib as gp

if __name__ == "__main__":
    gplvm = gp.NonlinearGPLVM()
    gp.pca(np.array([[1, 2, 3], [1, 2, 3]]), 2, True)
    help(gp.pca)
