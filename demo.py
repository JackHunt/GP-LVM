"""
BSD 3-Clause License

Copyright (c) 2022, Jack Miles Hunt
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
"""

import csv
import sys
import urllib.request
import os.path

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, './gplvm_lib')

import gplvm_lib as gp

IRIS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
IRIS_FNAME = 'iris.data'

@dataclass
class IrisData:
    features: np.array
    colours: list[str]

# Control plotting here.
SHOW_PLOTS = True
SAVE_PLOTS = False

def get_iris(use_colouring: bool = True) -> IrisData:
    """Loads the four dimensional Fisher Iris dataset.
    If the 'iris.data' file is not present in the working directory,
    this function attempts to download it.
    The last column of the dataset(the text labels) are ommitted.

    Args:
        use_colouring (bool, optional): Whether to assign colours to
        each class for plotting puposes. Defaults to True.

    Returns:
        IrisData: The Fisher Iris dataset.
    """

    iris = []
    colours = []
    if not os.path.isfile(IRIS_FNAME):
        print("Attempting to download the iris dataset.")
        try:
            urllib.request.urlretrieve(IRIS_URL, IRIS_FNAME)
        except urllib.request.URLError:
            sys.exit("Unable to download iris dataset. Quitting.")

    with open(IRIS_FNAME, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            if len(line):
                # Extract feature vector.
                iris.append(list(map(float, line[0:4])))

                # Extract class label and assign colour, if necessary.
                if use_colouring:
                    if line[4] == "Iris-setosa":
                        colours.append("red")
                    elif line[4] == "Iris-versicolor":
                        colours.append("green")
                    elif line[4] == "Iris-virginica":
                        colours.append("blue")
                    else:
                        sys.exit("Error reading class assignments. Check iris.data")

    # Randomise order - TO-DO: make this pythonic.
    for _ in range(0, 20):
        n = len(iris)
        A = np.random.randint(n, size=n)
        B = np.random.randint(n, size=n)
        for i in range(n):
            # Permute feature vectors.
            tmp = iris[A[i]]
            iris[A[i]] = iris[B[i]]
            iris[A[i]] = tmp

            # Permute colours.
            tmp = colours[A[i]]
            colours[A[i]] = colours[B[i]]
            colours[A[i]] = tmp

    return IrisData(np.asarray(iris), colours)

def plot(data: np.array,
         colours: list[str],
         dimensionality: int,
         title: str,
         method: str):
    """Helper function to reduce code duplication.
    """
    if dimensionality == 1:
        return gp.plot_1D(data, title, method, save_plot=SAVE_PLOTS)

    if dimensionality == 2:
        return gp.plot_2D(data, title, method, colours, save_plot=SAVE_PLOTS)

    if dimensionality == 3:
        return gp.plot_3D(data, title, method, colours, save_plot=SAVE_PLOTS)

    raise ValueError("Unsupported Dimensionality.")

def run_pca(data: IrisData,
            reduced_dimensions: int,
            show_scree: bool):
    """Runs standard PCA on the given dataset, optionally showing the associated
    Scree plot(normalised Eigenvalues)
    """
    print("-->Running PCA.")

    latent = gp.pca(data.features,
                    reduced_dimensions,
                    show_scree=show_scree,
                    save_scree=SAVE_PLOTS)

    plot(latent,
         data.colours,
         reduced_dimensions,
         "Iris Dataset",
         "PCA")

def run_linear_gplvm(data: IrisData,
                     reduced_dimensions: int,
                     beta: float):
    """Runs the Linear Gaussian Process Latent Variable Model on the given dataset.
    The resultant data plotted if the latent space is 1, 2 or 3 dimensional.
    """
    print("-->Running Linear GP-LVM.")

    gplvm = gp.LinearGPLVM(data.features)
    gplvm.compute(reduced_dimensions, beta)

    latent = gplvm.get_latent_space_representation()

    plot(latent,
         data.colours,
         reduced_dimensions,
         "Iris Dataset",
         "Linear GP-LVM")

def run_nonlinear_gplvm(data: IrisData,
                        reduced_dimensions: int):
    """Runs the Nonlinear Gaussian Process Latent Variable Model on the given dataset,
    for a given covariance matrix generating kernel.
    The resultant data plotted if the latent space is 1, 2 or 3 dimensional.
    """
    print("-->Running Nonlinear GP-LVM.")

    gplvm = gp.NonlinearGPLVM(data.features)
    gplvm.compute(reduced_dimensions,
                  50,
                  max_iterations=50,
                  jitter=4,
                  learn_rate=0.01,
                  momentum=0.01,
                  verbose=True)

    latent = gplvm.get_latent_space_representation()

    plot(latent,
         data.colours,
         reduced_dimensions,
         "Iris Dataset",
         "Nonlinear GP-LVM")

if __name__ == "__main__":
    # Dimension to reduce to.
    D = 2

    # Beta parameter for Linear GP-LVM.
    INITIAL_BETA = 2.0

    # Whether to display the Scree plot for PCA.
    SCREE = True

    ds = get_iris()

    run_pca(ds, D, SCREE)
    run_linear_gplvm(ds, D, INITIAL_BETA)
    run_nonlinear_gplvm(ds, D)

    if SHOW_PLOTS:
        plt.show()
