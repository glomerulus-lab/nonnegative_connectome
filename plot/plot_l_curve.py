import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import argparse
import glob
import math 
import copy 
from plot_l import plot 
import sys, inspect, os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from util.parse import parse
from util.sort import sort_vals




parser = argparse.ArgumentParser(description='Plot dominant factors of connectome solution')
# Arguments
parser.add_argument('testname',              type=str, nargs=1, help='Name of test to plot. "test", "top_view" or "flatmap".')
parser.add_argument('solution_name',         type=str, nargs=1, help='Name of .mat solution file, including or excluding file extension.')
parser.add_argument('path_to_solution',      type=str, nargs=1, help='Path to .mat solution file, excluding filename.')
parser.add_argument('title',                 type=str, nargs=1, help='Title of l-curve.')
# Flags
parser.add_argument('-nneg', action='store_true', help='Determines key used to get lambda value. When nonnegative solution used: nneg=True.')


## Plot the reg, loss l-curve for specified solutions
    # Input:
        # path: path to solution (str)
        # name: filename of solution (str)
        # title: figure title (str)
        # greedy: is greedy soltion (bool)
def create_l_curve(testname, path, name, title, greedy):
    # get data and sort it
    losses, regs, costs, lambs = parse(path+name, greedy)
    lambs_copy = copy.deepcopy(lambs)
    lambs_copy, regs = sort_vals(lambs_copy,regs)
    lambs, losses = sort_vals(lambs,losses)
    
    plt = plot(regs, losses, "Regularization", "Loss", title.replace("_", " "), lambs, r'$\lambda$')
    # save l-curve
    if(testname == 'test'):
        path = "../data/lambda_tests/"
    elif(testname == 'top_view'):
        path = "../data/lambda_tv/"
    elif(testname == 'flatmap'):
        path = "../data/lambda_fm/"

    plt.savefig(path+title.lower()+".jpg")
    plt.savefig(path+title.lower()+".svg")
    

if __name__ == '__main__':

    args = parser.parse_args()
    print('solution_name: ', args.solution_name[0], '\npath_to_solution: ', args.path_to_solution[0], '\ntitle: ', args.title[0])
    if(args.nneg):
        print('begin nonnegative l-curve')
        create_l_curve(args.testname[0], args.path_to_solution[0], args.solution_name[0], args.title[0], False)
    else:
        print('begin greedy l-curve')
        create_l_curve(args.testname[0], args.path_to_solution[0], args.solution_name[0], args.title[0], True)


