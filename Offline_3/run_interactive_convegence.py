from GMM_model import GaussianMixtureModel
from data_visualizer import DataVisualizerClass
from data_handler import load_dataset
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Script to run EM algorithm')

parser.add_argument('--data_file', type=str, default="data2D.txt", help="data file name")    
parser.add_argument('--K', type=int, default=5, help="number of components")
parser.add_argument('--max_iter', type=int, default=10, help="maximum iterations")

if __name__ == '__main__':
    
    args = parser.parse_args()
    dataset = args.data_file 
    K = args.K 
    X = load_dataset(dataset)
    max_iter = args.max_iter
    dv = DataVisualizerClass.getDVModel("icontour")
    GMM = GaussianMixtureModel(K, max_iter=max_iter, visualizer=dv)
    logl = GMM.fit(X)