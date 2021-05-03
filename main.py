#!/cluster/research-groups/harris/mcculls5/connectome_venv/bin/python
import argparse
import scipy.io
import numpy as np
import load_mat
import nonnegative_initialization
import nonnegative_connectome
import time

count = 0

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
parser.add_argument('-tol', type=float, default=1e-6, help="PGD stopping criteria tolerance")
parser.add_argument('-init_tol', type=float, default=1e-6, help="PGD stopping criteria tolerance for initialization refinement")
parser.add_argument('-alt_tol', type=float, default=1e-6, help="tolerance for alt_acc_prox_grad")


def balance_norms(Y, Z):
    print(Y.shape, Z.shape)
    rank = Y.shape[1]
    for r in range(rank):
        y_norm = np.linalg.norm(Y[:,r])
        z_norm = np.linalg.norm(Z[r,:])

        Y[:,r] *= np.sqrt(y_norm * z_norm) / y_norm
        Z[r,:] *= np.sqrt(y_norm * z_norm) / z_norm

    return Y, Z



if __name__ == '__main__':

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
    hp["lamb_reg"] = hp["lamb"] * (data["X"].shape[1] / data["Lx"].shape[0])  #(n_inj / n_x)

    cost_function = lambda W, H: nonnegative_connectome.regularized_cost(W, H.T, 
            data["X"], data["Y"], data["Lx"], data["Ly"], hp["lamb_reg"], data["Omega"])

    time_results["problem_setup"] = time.time() - start_time
    start_time = time.time()

    #Load greedy solution to initialize a nonnegative solution
    print("Loading greedy solution")
    Y, Z = load_mat.load_solution(hp["solution_name"], hp["from_lc"])

    # print("Y, Z norms", np.linalg.norm(Y, ord='fro'),np.linalg.norm(Z, ord='fro'))
    Y, Z = balance_norms(Y, Z)
    # print("Y, Z norms", np.linalg.norm(Y, ord='fro'),np.linalg.norm(Z, ord='fro'))


    time_results["load_greedy"] = time.time() - start_time

    # Get greedy cost
    greedy_cost = cost_function(Y, Z)
    print("Greedy solution cost:", greedy_cost)
    
    start_time = time.time()

    print("Initializing nonnegative solution")        
    W, H = nonnegative_initialization.init_nonnegative_factors(Y, Z)

    print("W, H init norms", np.linalg.norm(W, ord='fro'),np.linalg.norm(H, ord='fro'))

    time_results["initialization"] = time.time() - start_time

    # Get initialization cost
    nonneg_init_cost = cost_function(W, H)
    print("Nonnegative initialization cost:", nonneg_init_cost)

    start_time = time.time()

    # Refine nonnegative initialization bith alternating PGD 
    print("Refining nonnegative solution")
    W, H, init_costs = nonnegative_initialization.refine_nonnegative_factors(W, H, Y, Z,
                                    tol=hp["init_tol"], 
                                    alt_tol=hp["alt_tol"],
                                    max_outer_iter = hp["init_max_outer_iter"],
                                    max_inner_iter = hp["init_max_inner_iter"],
                                    max_line_iter = hp["init_max_line_iter"],
                                    calculate_cost = True)

    print("W, H final norms", np.linalg.norm(W, ord='fro'),np.linalg.norm(H, ord='fro'))

    time_results["refining"] = time.time() - start_time

    # Get refined cost
    refined_nonneg_cost = cost_function(np.array(W), np.array(H))
    print("Refined nonnegative cost:", refined_nonneg_cost)

    start_time = time.time()
    
    savedata = {"W":np.empty((2,1), dtype=object)}
    savedata["W"][0] = [W]
    savedata["W"][1] = [H.T]
    for key in hp.keys():
        savedata["hp_"+key] = hp[key]

    print("Saving refinied initialization")
    scipy.io.savemat("data/refined_init_"+hp["testname"]+"_"+hp["output_suffix"]+".mat", savedata)

    print("Starting nonnegative regression problem")
    U, V, costs = nonnegative_connectome.optimize_alt_pgd(W, H, 
                                    data["X"], data["Y"], data["Lx"], data["Ly"], data["Omega"],
                                    hp["lamb_reg"],
                                    tol=hp["tol"],
                                    alt_tol=hp["alt_tol"],
                                    max_outer_iter = hp["max_outer_iter"],
                                    max_inner_iter = hp["max_inner_iter"],
                                    max_line_iter = hp["max_line_iter"],
                                    calculate_cost = True)

    time_results["final_solution"] = time.time() - start_time     

    # Get refined cost
    final_nonneg_cost = cost_function(U, V)
    print("Final nonnegative cost:", final_nonneg_cost)

    # Save data to file
    print("Saving final solution with hyperparameters and experiment results...")
    data = {"W":np.empty((2,1), dtype=object)}
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

