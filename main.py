import argparse
import scipy.io
import numpy as np
import plot_test_heatmap
import plot_svectors
import plot_cost
import load_mat
import nonnegative_initialization
import nonnegative_connectome
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
def saveToMatFile(W, H, testname, suffix):
    
    data = {"W":np.empty((2,1), dtype=object)}
    data["W"][0] = [W]
    data["W"][1] = [np.transpose(H)]
    rank = W.shape[1]
    scipy.io.savemat("data/nonnegative_"+testname+"_"+suffix, data)
    



if __name__ == '__main__':

    args = parser.parse_args()
    testname = args.testname[0]
    hp["max_outer_iter"] = int(args.max_outer_iter[0])
    hp["max_inner_iter"] = int(args.max_inner_iter[0])
    hp["max_line_iter"]  = int(args.max_line_iter[0])

    data = load_mat.load_all_matricies_data(testname)
    if(testname=="top_view"):
        lamb = 1e6
    elif(testname=="flatmap"):
        lamb = 3e7
    else:
        lamb = 100

    # dimensionless regularization parameter
    lamb = lamb * (data["X"].shape[1] / data["Lx"].shape[0])  #(n_inj / n_x)


    print("Loading greedy solution")
    Y, Z = load_mat.load_solution(testname+"_solution", args.greedy)

    print("Greedy solution cost:", 
            nonnegative_connectome.regularized_cost(Y, Z.T, 
            data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]))

    print("Initializing nonnegative solution")
    W, H, costs = nonnegative_initialization.refined_init_nonnegative_factors(Y, Z, 
                                    max_outer_iter = hp["max_outer_iter"],
                                    max_inner_iter = hp["max_inner_iter"],
                                    max_line_iter = hp["max_line_iter"],
                                    calculate_cost = True)

    print("Nonnegative initialization cost:", 
        nonnegative_connectome.regularized_cost(W, H.T, 
        data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]))

    print("Saving refined initialization...")
    saveToMatFile(W, H, testname, "refined_init")

    print("Starting nonnegative regression problem")

    U, V, costs = nonnegative_connectome.optimize_alt_pgd(W, H, 
                                    data["X"], data["Y"], data["Lx"], data["Ly"], data["Omega"],
                                    lamb,
                                    max_outer_iter = hp["max_outer_iter"],
                                    max_inner_iter = hp["max_inner_iter"],
                                    max_line_iter = hp["max_line_iter"],
                                    calculate_cost = True)

    print("Saving final solution...")
    saveToMatFile(U, V, testname, "final")

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