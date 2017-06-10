#!/usr/bin/python3
import sys
sys.path.append('./gplvm_lib')

import linear_gplvm as lgp
import nonlinear_gplvm as nlgp

if __name__ == "__main__":
    gplvm = nlgp.NonlinearGPLVM()
    gplvm.hello()
