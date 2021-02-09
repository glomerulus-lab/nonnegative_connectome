import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
import palettable # https://jiffyclub.github.io/palettable/
iter_start = 1000

iters = ['10', '50', '100', '150', '200', '1000', '10000_5', '10000_6', '10000_7']
iterNums = [10, 50, 100, 150, 200, 1000, 10000, 10000, 10000]
cmap=palettable.colorbrewer.sequential.Reds_9.mpl_colormap
colors = [cmap(1.*i/len(iters)) for i in range(len(iters))]


def plot_all_costs(costs, names, filename):
    plt.figure()
    plt.clf()
    for i in range(len(costs)):
        plt.plot(np.arange(len(costs[i]))+iter_start, costs[i], color=colors[i], linestyle="solid", label=names[i])
    plt.legend(loc="best")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration by Inner Iters")
    plt.savefig(filename)
    plt.close()

def plot_times(iters, times, names, filename):
    plt.figure()
    plt.clf()
    for i in range(len(times)):
        plt.plot(iters[i], times[i], ".", color=colors[i], label=names[i])
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Time (Seconds)")
    plt.title("Time to Convergence vs Inner Iteration")
    plt.savefig(filename)
    plt.close()

def plot_final_cost(iters, costs, names, filename):
    plt.figure()
    plt.clf()
    plt.plot(iters, costs, 'g-')
    for i in range(len(costs)):
        plt.plot(iters[i], costs[i], ".", color=colors[i], label=names[i])
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Final Solution Cost")
    plt.title("Final Cost vs Inner Iterations")
    plt.savefig(filename)
    plt.close()

def plot_final_cost_time(times, costs, names, filename):
    plt.figure()
    plt.clf()

    plt.plot(times, costs, 'g-')
    for i in range(len(times)):
        plt.plot(times[i], costs[i], ".", color=colors[i], label=names[i])
    
    plt.legend(loc="best")
    plt.xlabel("Time(Seconds)")
    plt.ylabel("Final Cost")
    plt.title("Final Cost vs Time to convergence")
    plt.savefig(filename)
    plt.close()


costs = []
names = []
times = []
final_costs = []
for iter in iters:
    data = scipy.io.loadmat("data/nonnegative_test_100_init_q_test_"+str(iter)+".mat")
    costs.append(data["costs_pgd"][0][iter_start:])
    times.append(data["time_final_solution"][0] + data["time_refining"][0])
    print(data["time_final_solution"][0] +  data["time_refining"][0])
    names.append(str(iter)+" Iterations")
    final_costs.append(data["cost_final"][0])
print(len(costs))


plot_all_costs(costs, names, "plots/init_q_test_cost_plot")
plot_times(iterNums, times, names, "plots/init_q_test_time_plot")
plot_final_cost(iterNums, final_costs, names, "plots/init_q_test_final_costs_plot")
plot_final_cost_time(times, final_costs, names, "plots/init_q_test_time_final_costs_plot")