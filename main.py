import argparse
import scipy.io
import numpy as np
import plot_test_heatmap
import plot_svectors
import plot_cost
import load_mat
import nonnegative_initialization
import timeit



parser = argparse.ArgumentParser(description="Computes non-negative factors given greedy solution.")
# Arguments
parser.add_argument('testname',  type=str, nargs=1, help='Name of test to compute nonnegative factors')
parser.add_argument('max_outer_iter',  type=str, nargs=1, help='')
parser.add_argument('max_inner_iter',  type=str, nargs=1, help='')
parser.add_argument('max_line_iter',  type=str, nargs=1, help='')
# Flags
parser.add_argument('-greedy', action='store_true', help='Search ../lowrank_connectome/data for solution.')
parser.add_argument('-plot', dest='plot', action='store_true',help='Plot the solution as a heatmap using plot_test_heatmap.py')

# hyperperameters
hp = {
    'max_outer_iter': 10,
    'max_inner_iter': 10,
    'max_line_iter': 100
}

def hyperparameterStr():
    return '_'.join(str(x) for x in hp.values())


# Saves nonnegative factors W, H to 'nonnegative_<testname>_r_<rank>.mat'
# Expects W, H such that X = W @ H
# W should have shape (nx * r)
# H should have shape (r * ny)
# Where r is the rank of the nonnegative factors.
def saveToMatFile(testname, W, H):
    
    data = {"W":np.empty((2,1), dtype=object)}
    data["W"][0] = [W]
    data["W"][1] = [np.transpose(H)]
    rank = W.shape[1]
    scipy.io.savemat("data/nonnegative_"+testname+"_r_"+str(rank)+"_"+hyperparameterStr()+".mat", data)
    



if __name__ == '__main__':
    args = parser.parse_args()
    testname = args.testname[0]
    hp["max_outer_iter"] = int(args.max_outer_iter[0])
    hp["max_inner_iter"] = int(args.max_inner_iter[0])
    hp["max_line_iter"]  = int(args.max_line_iter[0])

    print("Loading greedy solution")
    Y, Z = load_mat.load_solution(testname+"_solution", args.greedy)

    print("Initializing nonnegative solution")
    W, H, costs = nonnegative_initialization.refined_init_nonnegative_factors(Y, Z, 
                                    max_outer_iter = hp["max_outer_iter"],
                                    max_inner_iter = hp["max_inner_iter"],
                                    max_line_iter = hp["max_line_iter"],
                                    calculate_cost = True)

    print("Saving data...")
    # saveToMatFile(testname, W_k, H_k)

    if(args.plot):
        print("Plotting...")
        if(testname == "test"):

            plot_test_heatmap.create_heatmap(Y, Z, "plots/test/greedy_solution")
            plot_test_heatmap.create_heatmap(W, H, "plots/test/nonneg_init")
            plot_test_heatmap.create_heatmap_test_truth("plots/test/test_truth")
            plot_cost.plot_1(costs,
                range(len(costs)),
                "Cost vs Iteration ("+hyperparameterStr()+")",
                "Cost",
                "Iteration",
                "plots/test/test_cost_"+hyperparameterStr())
        else:
            
            plot_svectors.plot_svectors(W, H, testname, "nonneg_" + testname + hyperparameterStr(),3)
            plot_cost.plot_1(costs,
                range(len(costs)),
                "Cost vs Iteration ("+hyperparameterStr()+")",
                "Cost",
                "Iteration",
                "plots/"+testname+"_cost_"+hyperparameterStr())
    print("Done")