import matplotlib.pyplot as plt
def plot_2(y1, y2, x, title, ylabel, xlabel, y1label, y2label, filename, xlog=False, yLog=False):
    plt.figure()
    plt.clf()

    plt.plot(x, y1, "b-", label=y1label)
    plt.plot(x, y2, "r-", label=y2label)
    if(xlog):
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

def plot_1(y, x, title, ylabel, xlabel, filename, xlog=False, yLog=False):
    plt.figure()
    plt.clf()

    plt.plot(x, y, "b-")
    if(xlog):
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_all_cost(y, max_x, y1, y2, filename):
    plt.figure()
    plt.clf()
    plt.plot(range(max_x), y, "b-")
    plt.plot([0,max_x], [y1, y1], ".r-",label="Greedy Cost")
    plt.plot([0,max_x], [y2, y2], ".r-",label="Refined Nonneg Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration compared to Initialization")
    plt.savefig(filename)
    plt.close()