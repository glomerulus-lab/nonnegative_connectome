import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import argparse
import glob
import collections 
import math 
import os
from plot_l import plot
import sys, inspect, os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import util.load_mat as load_mat
from util.sort import sort_vals





parser = argparse.ArgumentParser(description='Plot dominant factors of connectome solution')
# Arguments
parser.add_argument('solution_name',         type=str, nargs=1, help='Name of .mat solution file, including or excluding file extension.')
parser.add_argument('path_to_solution',    type=str, nargs=1, help='Path to .mat solution file, excluding filename.')
# Flags
parser.add_argument('-nneg', action='store_true', help='Determines key used to get lambda value. When nonnegative solution used: nneg=True.')


def load_w_matrices(filepath):
    file_true = '/home/stillwj3/Documents/research/lowrank_connectome/data/test_solution'
    data = sci.loadmat(filepath, variable_names='W_true')
    W_true = data['W_true']
    data = sci.loadmat(filepath, variable_names='W')
    W = data['W']

    return W_true, W

def plot_nneg_error(names,path):
    lambs = []
    errors = []
    for filepath in glob.glob(path+names):
        # print(filepath)
        filename = os.path.split(filepath)[1]
        lamb = load_mat.load_lamb(filename, path, greedy=False)
        lambs.append(lamb)
        W_true, W = load_w_matrices(filepath)
        error = calc_error(W_true, W)
        errors.append(error)

    # Order values by lambda
    lambs, errors = sort_vals(lambs, errors)

    # Plot lambda, error
    plt = plot_l(lambs, errors, r'$\lambda$',"Error","Nonnegative Error", lambs, r'$\lambda$')
    plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test_nneg_err.svg")
    plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test_nneg_err.jpg")

    

def plot_greedy_error(names,path):
    lambs = []
    errors = []
    for filepath in glob.glob(path+names):
        # print(filepath)
        filename = os.path.split(filepath)[1]
        lamb = load_mat.load_lamb(filename, path, greedy=True)
        lambs.append(lamb)
        W_true, W = load_w_matrices(filepath)
        error = calc_error(W_true, W)
        errors.append(error)
        
    # Order values by lambda
    lambs, errors = sort_vals(lambs, errors)
 
    # Plot lambda, error
    plt = plot_l(lambs, errors, r'$\lambda$',"Error","Greedy Error", lambs, r'$\lambda$')
    plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test_greedy_err.svg")
    plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test_greedy_err.jpg")


def calc_error(W_true, W):
    W = np.dot(W[0][0],W[1][0].T)
    D = W_true - W
    error = np.linalg.norm(D, ord='fro')
    return error


if __name__ == '__main__':
    args = parser.parse_args()
    if(args.nneg):
        print('begin nonnegative solution')
        print('solution_name: ', args.solution_name[0], ', path_to_solution: ', args.path_to_solution[0])
        plot_nneg_error(args.solution_name[0], args.path_to_solution[0])
    else:
        print('begin greedy solution')
        print('solution_name: ', args.solution_name[0], ', path_to_solution: ', args.path_to_solution[0])
        plot_greedy_error(args.solution_name[0], args.path_to_solution[0])

    
    # path = '/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/lambda_data/'
    # extension = 'test_lambda*.mat'

    # path = '/home/stillwj3/Documents/research/lowrank_connectome/matlab/solution/'
    # extension = 'lambda_test*.mat'