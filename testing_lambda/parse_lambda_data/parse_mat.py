import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import glob

# get lambda values used for greedy initialization
Lambdas= []
for filepath in glob.glob('/home/stillwj3/Documents/research/lowrank_connectome/matlab/lambda_tests*.mat'):
    data = sci.loadmat(filepath)
    Lambdas.append(data["Lambdas"][0])

# parse data for each lambda test
losses = []
regs = []
costs = []
#os.chdir('..')
for filepath in glob.glob('/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/lambda_data/test*.mat'):
    print(filepath)
    data = sci.loadmat(filepath)
    costs.append(data["cost_final"][0][0])
    losses.append(data["loss_final"][0][0])
    regs.append(data["reg_final"][0][0])

# plotting l-curve (reg, loss)
plt.plot(regs, losses, linestyle="-", color='b')
plt.scatter(regs, losses, 50, Lambdas)

# add colorbar and labels
cmap = mpl.cm.viridis
bounds = Lambdas[0]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label="Lambda")
plt.xlabel("Regularization")
plt.ylabel("Loss")
# plt.show()

# save l-curve
plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test-l-curve.svg")
plt.savefig("/home/stillwj3/Documents/research/nonnegative_connectome/data/lambda_tests/test-l-curve.jpg")


