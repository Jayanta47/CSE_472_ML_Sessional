import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal

colors = ['blue', 'green', 'red', 'black', 'magenta', 'yellow', 'violet', 'white']

class DataVisualizerClass:
    def __init__(self) -> None:
         pass 

    def getDVModel(modelName):
        if modelName == "2Dplot":
            return TwoDPlotDataVisualizer()
        elif modelName == "icontour":
            return ContourInteractiveVisualizer()
        else:
            return None


class TwoDPlotDataVisualizer:
    def __init__(self) -> None:
        pass

    def visualize(self, X, **kwargs):
        legend_x = kwargs.get("legend_x", 0.5)
        legend_y = kwargs.get("legend_y", 0.5)
        plt.figure(figsize=(10, 10))
        plt.plot(range(len(X)), X)
        plt.xlabel(legend_x)
        plt.ylabel(legend_y)
        # plt.savefig("log_likelihood.png")
        plt.show()

class ContourInteractiveVisualizer:
    def __init__(self) -> None:
        plt.figure(figsize=(25, 10), num=1)

    def visualize(self, X, **kwargs):
        # legend_x = kwargs.get("legend_x", 0.5)
        # legend_y = kwargs.get("legend_y", 0.5)
        n_components = kwargs.get("n_components")
        mean = kwargs.get("mean")
        cov = kwargs.get("cov")

        plt.ion()
        # plt.figure(figsize=(10, 10))
        # ax0 = fig.add_subplot(111)
        # Plot the data points
        plt.scatter(X[:, 0], X[:, 1], s=10, edgecolor='k', alpha=0.5, marker='o')
        
        #Create grid of points
        x_grid, y_grid = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):.01, np.min(X[:, 1]):np.max(X[:, 1]):.01]
        positions = np.dstack((x_grid, y_grid))

        
        for k in range(n_components):
            rv = multivariate_normal(mean[k], cov[k], allow_singular=True).pdf(positions)
            plt.contour(x_grid, y_grid, rv, colors=colors[k%8], alpha=0.8, linewidths=2)
        plt.title("GMM-EM: " + str(n_components) + " Gaussians", size=15)
        
        plt.pause(0.05)
        plt.clf()
        plt.show()
        plt.ioff()