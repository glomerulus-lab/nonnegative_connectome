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
parser.add_argument('solution_name',         type=str, nargs=1, help='Name of .mat solution file, including or excluding file extension.')
parser.add_argument('path_to_solution',      type=str, nargs=1, help='Path to .mat solution file, excluding filename.')
parser.add_argument('title',                 type=str, nargs=1, help='Title of l-curve.')
# Flags
parser.add_argument('-nneg', action='store_true', help='Determines key used to get lambda value. When nonnegative solution used: nneg=True.')

def create_l_curve(path, name, title, greedy):
    losses, regs, costs, lambs = parse(path+name, greedy)
    lambs_copy = copy.deepcopy(lambs)
    lambs_copy, regs = sort_vals(lambs_copy,regs)
    lambs, losses = sort_vals(lambs,losses)

    for x in range(len(lambs)):
        if(greedy):
            lambs[x]=math.log10(lambs[x])
    
    plt = plot(regs, losses, "Regularization", "Loss", title.replace("_", " "), lambs, r'$\lambda$')
    # plt.show()

    # save l-curve
    plt.savefig(path+"../"+title.lower()+".svg")
    plt.savefig(path+"../"+title.lower()+".jpg")

if __name__ == '__main__':

    args = parser.parse_args()
    print('solution_name: ', args.solution_name[0], '\npath_to_solution: ', args.path_to_solution[0], '\ntitle: ', args.title[0])
    if(args.nneg):
        print('begin nonnegative l-curve')
        create_l_curve(args.path_to_solution[0], args.solution_name[0], args.title[0], False)
    else:
        print('begin greedy l-curve')
        create_l_curve(args.path_to_solution[0], args.solution_name[0], args.title[0], True)
        
    # '/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/lambda_data/test*.mat'
    # '/home/stillwj3/Documents/research/lowrank_connectome/matlab/lambda_fm_final.mat'
    # '/home/stillwj3/Documents/research/lowrank_connectome/matlab/solution/
    # 'lambda_test*.mat'
    # 'lambda_tv*.mat'

