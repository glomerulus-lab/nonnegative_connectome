import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl

losses = []
regs = []
costs = []
lambdas = [-2,-1,0,1,2,3,4,5,6,7,8,9,10]
# parse data for each lambda test
for x in range(-2,11):
    # Create filename
    if x != 0:
        lamb = "1e"+str(x)
    else:
        lamb = "0"
    filename = "test_" + lamb

    # Load and save data
    data = sci.loadmat("lambda_tests/lambda_data/"+filename, appendmat=True)
    costs.append(data["cost_final"][0][0])
    losses.append(data["loss_final"][0][0])
    regs.append(data["reg_final"][0][0])

# print(regs)
# print(losses)

# plotting l-curve (reg, loss)
plt.plot(regs, losses, linestyle="-", color='b')
plt.scatter(regs, losses, 50, lambdas)

# add colorbar and labels
cmap = mpl.cm.viridis
bounds = lambdas
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label="Lambda")
plt.xlabel("Regularization")
plt.ylabel("Loss")
# plt.show()

# save l-curve
plt.savefig("lambda_tests/1e-2_nonneg-l-curve.svg")
plt.savefig("lambda_tests/1e-2_nonneg-l-curve.jpg")


