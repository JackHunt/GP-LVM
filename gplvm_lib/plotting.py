import numpy as np
import matplotlib.pyplot as plt

def plot1D(data, title, method, savePlot = False):
    if data.shape[0] == 0 or data.shape[1] > 1:
        raise ValueError("Incorrect dimensions of data for 1D plotting.")

    #Plot.
    plt.figure()
    x = np.array(range(1, data.shape[0]+1))
    plt.plot(x, data)
    plt.xticks(x)
    plt.xlabel("Data point ID's")
    plt.ylabel("Value")
    plt.title("1D Latent Space Representation of %s using %s" % (title, method))
    plt.grid(True)
    
    if savePlot:
        plt.savefig("1D_latent_%s_%s.png" % (title.replace(" ", ""), method.replace(" ", "")))
    #plt.show()
    
def plot2D(data, title, method, savePlot = False):
    if data.shape[0] == 0 or data.shape[1] > 2:
        raise ValueError("Incorrect dimensions of data for 2D plotting.")
        
    x = data[:, 0]
    y = data[:, 1]
        
    #Plot
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("X Dimension")
    plt.ylabel("Y Dimension")
    plt.title("2D Latent Space Representation of %s using %s" % (title, method))
    plt.grid(True)
    
    if savePlot:
        plt.savefig("2D_latent_%s_%s.png" % (title.replace(" ", ""), method.replace(" ", "")))
    #plt.show()
    
def plot3D(data, title, method, savePlot = False):
    pass