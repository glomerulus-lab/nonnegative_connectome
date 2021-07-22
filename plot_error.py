import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import argparse
import glob
import collections 
import math 
import os
import util.load_mat as load_mat

parser = argparse.ArgumentParser(description='Plot dominant factors of connectome solution')
# Arguments
parser.add_argument('solution_name',         type=str, nargs=1, help='Name of .mat solution file, including or excluding file extension.')
parser.add_argument('path_to_solution',    type=str, nargs=1, help='Path to .mat solution file, excluding filename.')
# Flags
parser.add_argument('-nneg', action='store_true', help='Determines key used to get lambda value. When nonnegative solution used: nneg=True.')


def load_W_matrices(filepath):
    file_true = '/home/stillwj3/Documents/research/lowrank_connectome/matlab/lambda_tests.mat'
    data = sci.loadmat(file_true, variable_names='W_true')
    W_true = data['W_true']
    data = sci.loadmat(filepath, variable_names='W')
    W = data['W']

    return W_true, W

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
    else:
        print('begin greedy solution')
    # path = '/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/lambda_data/'
    # extension = 'test_lambda*.mat'

    # path = '/home/stillwj3/Documents/research/lowrank_connectome/solution/'
    # extension = 'lambda_test*.mat'

