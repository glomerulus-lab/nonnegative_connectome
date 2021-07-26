import numpy as np
import collections 
import math 


def sort_vals(key, vals):
    dict_sort = {}
    for x in range(len(key)):
        key[x] = math.log10(key[x])
        dict_sort[key[x]]= [vals[x]]  
    dict_sort = collections.OrderedDict(sorted(dict_sort.items()))
    key.sort()
    for x in range(len(key)):
        vals[x] = dict_sort[key[x]][0]
    return key, vals
