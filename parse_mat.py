import argparse
import scipy.io as sci
import numpy as np

# parser = argparse.ArgumentParser(description="Gets data")

# # Arguments
# parser.add_argument('filename',  type=str, help='Name of mat file to parse')
data_list = []
losses = []
regs = []
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

import matplotlib.pyplot as plt 

# plotting the points  
plt.plot(regs, losses, linestyle="-", marker=".", color='b')
plt.xlabel("Regularization")
plt.ylabel("Loss")
plt.show()
