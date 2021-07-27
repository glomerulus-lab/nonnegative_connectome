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

## Given a filepath retrieve estimated W and true W
    # Input: filepath = path+file (str)
    # Output: W, W_true (matrices)
def load_w_matrices(filepath):
    file_true = '/home/stillwj3/Documents/research/lowrank_connectome/data/test_solution'
    data = sci.loadmat(file_true, variable_names='W_true')
    W_true = data['W_true']
    data = sci.loadmat(filepath, variable_names='W')
    W = data['W']

    return W_true, W
def get_key_val(filepath, var):
    data = sci.loadmat(filepath, variable_names=var)
    val = data[var]

## Plot the error between true and estimated for nonnegative solutions
    # Input: names: 
        # name pointing to solution(s), Ex. test*.mat (str)
        # path: a string directory path, Ex. path/to/solution/ (str)
    # Output: An error curve
def plot_nneg_error(names,path):
    lambs = []
    errors = []
    
    for filepath in glob.glob(path+names):
        data = sci.loadmat(filepath, variable_names='hp_lamb')
        lamb = data['hp_lamb'][0][0]
        lambs.append(lamb)
        W_true, W = load_w_matrices(filepath)
        error = calc_error(W_true, W)
        errors.append(error)

    # Order values by lambda
    lambs, errors = sort_vals(lambs, errors)

    # Plot lambda, error
    plt = plot(lambs, errors, r'$\lambda$',"Error","Nonnegative Error", lambs, r'$\lambda$')
    plt.savefig("../data/lambda_tests/test_nneg_err.svg")
    plt.savefig("../data/lambda_tests/test_nneg_err.jpg")

## Plot the error between true and estimated for greedy solutions    
    # Input:
        # names: name pointing to solution(s), Ex. test*.mat (str)
        # path: a string directory path, Ex. path/to/solution/ (str)
    # Output: An error curve
def plot_greedy_error(names,path):
    lambs = []
    errors = []
    for filepath in glob.glob(path+names):
        lamb = sci.loadmat(filepath, variable_names='lamb')
        lamb = data['lamb'][0][0]
        lambs.append(lamb)
        W_true, W = load_w_matrices(filepath)
        error = calc_error(W_true, W)
        errors.append(error)
        
    # Order values by lambda
    lambs, errors = sort_vals(lambs, errors)
 
    # Plot lambda, error
    plt = plot(lambs, errors, r'$\lambda$',"Error","Greedy Error", lambs, r'$\lambda$')
    plt.savefig("../data/lambda_tests/test_greedy_err.svg")
    plt.savefig("../data/lambda_tests/test_greedy_err.jpg")

## Calculate error between W_true and W using frobenius norm
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
