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

import argparse
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
    """A simple dataclass consisting of iris features and colour labels.
    """
    features: np.array
    colours: list[str]

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

    def _label_as_colour(label: str) -> str:
        if label == "Iris-setosa":
            return "red"

        if label == "Iris-versicolor":
            return "green"

        if label == "Iris-virginica":
            return "blue"

        raise ValueError("Error reading class assignments. Check iris.data")

    with open(IRIS_FNAME, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            if len(line):
                # Extract feature vector.
                iris.append(list(map(float, line[0:4])))

                # Extract class label and assign colour, if necessary.
                if use_colouring:
                    colours.append(_label_as_colour(line[4]))

    # Randomise order.
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
    """Plots the latent space representation of the data.

    Args:
        data (np.array): Iris features.
        colours (list[str]): Class colour labels.
        dimensionality (int): Dimensionality of the plot.
        title (str): Title of the plot.
        method (str): Name of the model.

    Raises:
        ValueError: If an unsupported dimensionality is provided.
    """
    if dimensionality == 1:
        gp.plot_1D(data, title, method, save_plot=False)

    if dimensionality == 2:
        gp.plot_2D(data, title, method, colours, save_plot=False)

    if dimensionality == 3:
        gp.plot_3D(data, title, method, colours, save_plot=False)

    raise ValueError("Unsupported Dimensionality.")

def run_pca(data: IrisData,
            reduced_dimensions: int = 2,
            show_scree: bool = True):
    """Runs standard PCA on the Iris dataset.

    Args:
        data (IrisData): The Iris dataset on which to run PCA.
        reduced_dimensions (int): Target dimensionality.
          Default is 2.
        show_scree (bool): Whether to show a plot of normalised eigenvalues.
          Default is True.
    """
    print("-->Running PCA.")

    latent = gp.pca(data.features,
                    reduced_dimensions,
                    show_scree=show_scree,
                    save_scree=False)

    plot(latent,
         data.colours,
         reduced_dimensions,
         "Iris Dataset",
         "PCA")

def run_linear_gplvm(data: IrisData,
                     reduced_dimensions: int = 2,
                     beta: float = 2.0):
    """Runs the Linear Gaussian Process Latent Variable Model
       on the Iris dataset.

    Args:
        data (IrisData): The Iris dataset on which to run the Linear GPLVM.
        reduced_dimensions (int): Target dimensionality. Default is 2.
        beta (float): The Linear GPLVM Regularizer, beta. Default is 2.0.
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
                        reduced_dimensions: int = 2,
                        batch_size: int = 25,
                        max_iterations: int = 50,
                        jitter: int = 4,
                        learning_rate: float = 0.01,
                        momentum: float = 0.01):
    """Runs the Nonlinear Gaussian Process Latent Variable Model on
       the Iris dataset, for a given covariance matrix generating kernel.

    Args:
        data (IrisData): The Iris dataset on which to run the Nonlinear GPLVM.
        reduced_dimensions (int): Target dimensionality.
            Default is 2.
        max_iterations (int): Upper bound on optimizer epochs.
            Default is 50.
        jitter (int): GP jitter factor.
            Default is 4.
        learning_rate (float): Optimizer learning rate/scale factor.
            Default is 0.01.
        momentum (float): Optimizer momentum coefficient.
            Default is 0.01.
    """
    print("-->Running Nonlinear GP-LVM.")

    gplvm = gp.NonlinearGPLVM(data.features)
    gplvm.compute(reduced_dimensions,
                  batch_size,
                  max_iterations=max_iterations,
                  jitter=jitter,
                  learn_rate=learning_rate,
                  momentum=momentum,
                  verbose=True)

    latent = gplvm.get_latent_space_representation()

    plot(latent,
         data.colours,
         reduced_dimensions,
         "Iris Dataset",
         "Nonlinear GP-LVM")

if __name__ == "__main__":
    # Take all config items as optional params.
    parser = argparse.ArgumentParser(description='GPLVM Example.')
    parser.add_argument('--reduced_dims', type=int, default=2)
    parser.add_argument('--beta_regularizer', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.01)
    parser.add_argument('--jitter', type=int, default=4)

    # Get config options for the various models.
    args = parser.parse_args()
    reduced_dims = args.reduced_dims

    # Get the data and run the models.
    ds = get_iris()

    run_pca(ds, reduced_dimensions=reduced_dims)

    run_linear_gplvm(ds,
                     reduced_dimensions=reduced_dims,
                     beta=args.beta_regularizer)

    run_nonlinear_gplvm(ds,
                        reduced_dimensions=reduced_dims,
                        batch_size=args.batch_size,
                        max_iterations=args.max_iterations,
                        jitter=args.jitter,
                        learning_rate=args.learning_rate,
                        momentum=args.momentum)
    
    # Finally, plot.
    plt.show()
