# file to implement the driver to run EM algorithm
from data_handler import load_dataset
from GMM_model import GaussianMixtureModel
from data_visualizer import DataVisualizerClass
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Script to run EM algorithm')

parser.add_argument('--data_file', type=str, default="data2D.txt", help="data file name")    
parser.add_argument('--k', type=int, default=10, help="range of Gaussian distributions")
# parser.add_argument('--num_iter', type=int, default=3000, help="number of iterations")
parser.add_argument('--verbose', type=bool, default=False, help="verbose")

if __name__ == '__main__':

    args = parser.parse_args()
    dataset = args.data_file 
    K = args.k 
    verbose = args.verbose
    X = load_dataset(dataset)
    log_likelihood = []


    K_star = -1;
    for k in range(K):
        GMM = GaussianMixtureModel(k+1)
        logl = GMM.fit(X)
        log_likelihood.append(logl)
        if k>0 and K_star == -1:
            prev_log_l = log_likelihood[k-1]
            curr_log_l = log_likelihood[k]
            if np.abs(curr_log_l - prev_log_l)/np.abs(prev_log_l) < 1e-2:
                K_star = k+1
    print("Converged at K = {}".format(K_star))
    print(log_likelihood)

    dv = DataVisualizerClass.getDVModel("2Dplot")

    dv.visualize(log_likelihood, legend_x="K", legend_y="Log Likelihood")
    

    


    

