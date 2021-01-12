import argparse
import scipy.io
import numpy as np

import plot_svectors
import plot_test_heatmap

import plot_cost
import load_mat
import nonnegative_factorization as nf
import timeit
import os

parser = argparse.ArgumentParser(description='Computes non-negative factors given greedy solution.')
# Arguments
parser.add_argument('testname',  type=str, nargs=1, help='Name of test to compute nonnegative factors')

#Flags
parser.add_argument('-greedy', action='store_true', help='Search ../lowrank_connectome/data for solution.')
parser.add_argument('-plot', dest='plot', action='store_true', help='Plot the solution using plot_svectors.py or plot_test_heatmap.py')

# Hyperparameters
hp = {
    'max_outer_iter': 50,
    'max_inner_iter': 50,
    'max_line_iter':100,
    'h_pow': -3
}

#converts hyperparameter configuration to string for output naming
def hyperparameterStr():
    result = '_'.join(str(int(x)) for x in hp.values())
    return result

hyperparameterStr()

#Saves factors to .mat format consistent with lowrank_connectome.
def save_solution(testname, W, H):
    if not os.path.exists('data'):
        os.makedirs('data')   
    data = {"W":np.empty((2,1), dtype=object)}
    data["W"][0] = [W]
    data["W"][1] = [np.transpose(H)]
    rank = W.shape[1]
    filename = "data/"+testname+"_r_"+str(rank)+"_"+hyperparameterStr()+".mat"
    scipy.io.savemat(filename, data)
    print("Saved as", filename)


if __name__ == '__main__':
    args = parser.parse_args()
    testname = args.testname[0]

    print("Loading greedy solution.")
    Y, Z = load_mat.load_solution(testname+'_solution', args.greedy)
    
    print("Initializing nonnegative solution.")
    W, H = nf.initialize_nonnegative_factors(Y, Z)

    print("Starting Alternating PGD.")
    W_k, H_k, costs = nf.alternating_pgd(Y, Z, W, H, 
                                nf.frobenius_squared_trace,
                                nf.grad_frobenius_squared,
                                nf.indicator_positive,
                                nf.proj_nonnegative,
                                max_outer_iter = hp['max_outer_iter'],
                                max_inner_iter = hp['max_inner_iter'],
                                max_line_iter = hp['max_line_iter'])

    print("Saving nonnegative_solution.")
    save_solution(testname, W_k, H_k)


    if(args.plot):
        if(testname == "test"):
            print("Plotting test.")


            

            plot_test_heatmap.create_heatmap(W, H, 'plots/test/test_nonneg_init')

            plot_test_heatmap.create_heatmap(W_k, H_k, 'plots/test/test_nonneg_'+hyperparameterStr())
            plot_cost.plot_1(costs,
                range(len(costs)),
                'Cost vs Iteration ('+hyperparameterStr()+')',
                'Cost',
                'Iteration',
                'plots/test/test_cost_'+hyperparameterStr())
            
            plot_test_heatmap.create_heatmap_test_truth('plots/test/test_truth')

            plot_test_heatmap._create_heatmap_from_solution('test_solution', 'plots/test/test_greedy_solution')

        else:
            print("Plotting nonnegative initialization.")
            plot_svectors.plot_svectors(W, H, testname, testname+'_nonneg_init')

            print("Plotting nonnegative solution.")
            plot_svectors.plot_svectors(W_k, H_k, testname, testname+'_nonneg_'+hyperparameterStr())

            print("Plotting PGD costs.")
            plot_cost.plot_1(costs,
                range(len(costs)),
                'Cost vs Iteration ('+hyperparameterStr()+')',
                'Cost',
                'Iteration',
                'plots/'+args.testname[0]+'_cost_'+hyperparameterStr())