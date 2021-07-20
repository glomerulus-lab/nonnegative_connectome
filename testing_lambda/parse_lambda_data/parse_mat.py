import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import glob
import collections 
import math  

# parse data for each lambda test
losses = []
regs = []
costs = []
Lambdas = []
for filepath in glob.glob('/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/lambda_data/test*.mat'):
    data = sci.loadmat(filepath)
    costs.append(data["cost_final"][0][0])
    losses.append(data["loss_final"][0][0])
    regs.append(data["reg_final"][0][0])
    Lambdas.append(data["hp_lamb"][0][0])

# Arrange values so test data is sorted by lambda (ascending)
# Lambs: dict stores cost, loss, reg, with lambda = key
Lambs = {}
for x in range(len(Lambdas)):
    Lambs[Lambdas[x]]= [costs[x],losses[x],regs[x]]    
Lambs = collections.OrderedDict(sorted(Lambs.items()))

# Sort list of lambdas (ascending) and replace values in... 
# ...costs, losses, regs with corresponding value in dict so the order of tests matches
Lambdas.sort()
for x in range(len(Lambdas)):
    costs[x] = Lambs[Lambdas[x]][0]
    losses[x] = Lambs[Lambdas[x]][1]
    regs[x] = Lambs[Lambdas[x]][2]

# Store lambda as log_10 value
for x in range(len(Lambdas)):
    Lambdas[x] = math.log10(Lambdas[x])



# plotting l-curve (reg, loss)
plt.plot(regs, losses, linestyle="-", color='b')
plt.scatter(regs, losses, 50, Lambdas)

# add colorbar and labels
cmap = mpl.cm.viridis
bounds = Lambdas
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label="Lambda")
plt.xlabel("Regularization")
plt.ylabel("Loss")
# plt.show()

# save l-curve
plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test-l-curve.svg")
plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test-l-curve.jpg")


