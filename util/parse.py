import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import glob
import collections 
import math 
import copy 



## Parse solution(s) and save loss, regularization, cost, and lambda values in lists
    # Input:
        # filename: name of file to parse (includes full path) (str)
        # greedy: is greedy solution (bool)
    # Output: lists for each value
def parse(filename,greedy=True):
    # parse data for each lambda test
    losses = []
    regs = []
    costs = []
    lambs = []
    for filepath in glob.glob(filename):
        if(greedy):
            list_variables= ["cost","loss","reg","lamb"]
        else:
            list_variables= ["cost_final", "loss_final", "reg_final", "hp_lamb"]
        
        data = sci.loadmat(filepath, variable_names=list_variables[0])
        costs.append(data[list_variables[0]][0][0])
        data = sci.loadmat(filepath, variable_names=list_variables[1])
        losses.append(data[list_variables[1]][0][0])
        data = sci.loadmat(filepath, variable_names=list_variables[2])
        regs.append(data[list_variables[2]][0][0])
        data = sci.loadmat(filepath, variable_names=list_variables[3])
        lambs.append(data[list_variables[3]][0][0])

    return losses,regs,costs,lambs