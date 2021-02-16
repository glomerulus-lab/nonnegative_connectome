import argparse
import scipy.io
import jax.numpy as np
import numpy
import load_mat
import nonnegative_initialization
import nonnegative_connectome
import time

from jax.config import config
config.update('jax_enable_x64', True)


parser = argparse.ArgumentParser(description="Computes non-negative factors given greedy solution.")
# Arguments
parser.add_argument('testname',  type=str, help='Name of test to compute nonnegative factors')
parser.add_argument('solution_name', type=str, help='Name of greedy solution to initialize with,')
parser.add_argument('output_suffix',  type=str, help='')
parser.add_argument('init_max_outer_iter',  type=int, help='')
parser.add_argument('init_max_inner_iter',  type=int, help='')
parser.add_argument('init_max_line_iter',  type=int,help='')
parser.add_argument('max_outer_iter',  type=int, help='')
parser.add_argument('max_inner_iter',  type=int, help='')
parser.add_argument('max_line_iter',  type=int, help='')
# Flags
parser.add_argument('-from_lc', action='store_true', help='Search ../lowrank_connectome/data for solution.')
parser.add_argument('-init_tol', type=float, default=1e-7, help="PGD stopping criteria tolerance for initialization refinement")
parser.add_argument('-tol', type=float, default=1e-7, help="PGD stopping criteria tolerance")

  
def main():


    # Parse arguments
    hp = vars(parser.parse_args())
    
    time_results = {}
    start_time = time.time()
    # Load data for problem setup
    data = load_mat.load_all_matricies(hp["testname"])


    #Use fixed values for lambda for consistency
    if(hp["testname"]=="top_view"):
        hp["lamb"] = 1e6
    elif(hp["testname"]=="flatmap"):
        hp["lamb"] = 3e7
    else:
        hp["lamb"] = 100

    # make regularization parameter dimensionless
    hp["lamb_reg"] = hp["lamb"] * (data["X"].shape[1] / data["X"].shape[0])  #(n_inj / n_x)
    
    cost_function = lambda W, H: nonnegative_connectome.regularized_cost( W, H.T,
        data["X"], 
        data["Y"], 
        data["Lx"], 
        data["Ly"], 
        hp["lamb_reg"], 
        data["Omega"])

    time_results["problem_setup"] = time.time() - start_time
    start_time = time.time()

    #Load greedy solution to initialize a nonnegative solution
    print("Loading greedy solution")
    Y, Z = load_mat.load_solution(hp["solution_name"], hp["from_lc"])
    print("Y", Y.shape)
    time_results["load_greedy"] = time.time() - start_time

    # Get greedy cost
    greedy_cost = cost_function(Y, Z)
    print("Greedy solution cost:", greedy_cost)
    
    start_time = time.time()

    print("Initializing nonnegative solution")        
    W, H = nonnegative_initialization.init_nonnegative_factors(Y, Z)
    print("W", W.shape)

    time_results["initialization"] = time.time() - start_time

    # Get initialization cost
    nonneg_init_cost = cost_function(W, H)
    print("Nonnegative initialization cost:", nonneg_init_cost)

    start_time = time.time()

    # Refine nonnegative initialization bith alternating PGD 
    print("Refining nonnegative solution")
    W, H, init_costs = nonnegative_initialization.refine_nonnegative_factors(W, H, Y, Z,
                                    tol=hp["init_tol"], 
                                    max_outer_iter = hp["init_max_outer_iter"],
                                    max_inner_iter = hp["init_max_inner_iter"],
                                    max_line_iter = hp["init_max_line_iter"],
                                    calculate_cost = True)
    print("W", W.shape)
    time_results["refining"] = time.time() - start_time

    # Get refined cost
    refined_nonneg_cost = cost_function(W, H)
    print("Refined nonnegative cost:", refined_nonneg_cost)

    start_time = time.time()

    print("Starting nonnegative regression problem")
    U, V, costs = nonnegative_connectome.optimize_alt_pgd(W, H, 
                                    data["X"], data["Y"], data["Lx"], data["Ly"], data["Omega"],
                                    hp["lamb_reg"],
                                    tol=hp["tol"],
                                    max_outer_iter = hp["max_outer_iter"],
                                    max_inner_iter = hp["max_inner_iter"],
                                    max_line_iter = hp["max_line_iter"],
                                    calculate_cost = True)
    print("U", W.shape)
    time_results["final_solution"] = time.time() - start_time     

    # Get refined cost
    final_nonneg_cost = cost_function(U, V)
    print("Final nonnegative cost:", final_nonneg_cost)

    # Save data to file
    print("Saving final solution with hyperparameters and experiment results...")
    data = {"W":numpy.empty((2,1), dtype=object)}
    data["W"][0] = [U]
    data["W"][1] = [V.T]
    data["costs_init_pgd"] = init_costs
    data["costs_pgd"] = costs
    data["cost_greedy"] = greedy_cost
    data["cost_init"] = nonneg_init_cost
    data["cost_refined"] = refined_nonneg_cost
    data["cost_final"] = final_nonneg_cost
    for key in hp.keys():
        data["hp_"+key] = hp[key]
    
    for key in time_results.keys():
        data["time_"+key] = time_results[key]

    scipy.io.savemat("data/nonnegative_"+hp["testname"]+"_"+hp["output_suffix"]+".mat", data)
    print("Done")
    print(hp)
    print(time_results)


if __name__ == '__main__':
    main()