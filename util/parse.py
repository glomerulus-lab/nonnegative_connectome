import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import glob
import collections 
import math 
import copy 




def parse(filename,greedy=True):
    # parse data for each lambda test
    losses = []
    regs = []
    costs = []
    lambs = []
    print(filename)
    for filepath in glob.glob(filename):
        print(filepath)
        if(greedy):
            list_variables= ["cost","loss","reg","lamb"]
        else:
            list_variables= ["cost_final", "loss_final", "reg_final", "hp_lamb"]
        list_variables[0]
        data = sci.loadmat(filepath, variable_names=list_variables[0])
        costs.append(data[list_variables[0]][0][0])
        data = sci.loadmat(filepath, variable_names=list_variables[1])
        losses.append(data[list_variables[1]][0][0])
        data = sci.loadmat(filepath, variable_names=list_variables[2])
        regs.append(data[list_variables[2]][0][0])
        data = sci.loadmat(filepath, variable_names=list_variables[3])
        lambs.append(data[list_variables[3]][0][0])

    return losses,regs,costs,lambs