import numpy as np
import collections 
import math 

## Sort key and vals by order of key, so values remain paired by index
    # Input:
        # key: key values of dictionary (list)
        # vals: value at some key, k (list)
def sort_vals(key, vals):
    dict_sort = {}
    # for each value in key, store a corresponding value from vals in a dict
    for x in range(len(key)):
        key[x] = math.log10(key[x])
        dict_sort[key[x]]= [vals[x]] 

    # sort dict by order of keys, and sort key in ascending order 
    dict_sort = collections.OrderedDict(sorted(dict_sort.items()))
    key.sort()

    # replace original values in vals with sorted values
    for x in range(len(key)):
        vals[x] = dict_sort[key[x]][0]
    return key, vals
