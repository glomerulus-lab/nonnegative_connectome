import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt 

data_list = []
losses = []
regs = []

# parse data for each lambda test
for x in range(0,11):
    if x != 0:
        lamb = "1e"+str(x)
    else:
        lamb = "0"
    filename = "nonnegative_test_" + lamb
    data = sci.loadmat("lambda_tests/lambda_data/"+filename, appendmat=True)
    data_list.append(data)
    losses.append(data["loss_final"][0][0])
    regs.append(data["reg_final"][0][0])

# plotting l-curve (reg, loss)
plt.plot(regs, losses, linestyle="-", marker=".", color='b')
plt.xlabel("Regularization")
plt.ylabel("Loss")
plt.show()

