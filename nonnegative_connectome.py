import numpy as np
from typing import Callable
import alt_acc_prox_grad
import load_mat


def optimize_alt_pgd(U, V,
                    testname,
                    lamb, 
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost=False):

    data = load_mat.load_all_matricies_data(testname)
    # data["Omega"] = data["Omega"].astype(np.int)

    print("Performing PDG on regularized nonnegative regression problem.")
    U, V, costs = alt_acc_prox_grad.alternating_pgd(
                    U, V, 
                    lambda U, V: regularized_cost(U, V.T, data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]),
                    lambda U, V: grad_cost_U(U, V, data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]),
                    lambda U, V: grad_cost_V(U, V, data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]),
                    indicator_positive,
                    proj_nonnegative,
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost)
    return U, V, costs


def P_Omega(X, Omega):
    print("@                              P_Omega")
    return X - np.multiply(X, Omega)

def grad_cost_U(U, V, X, Y, Lx, Ly, lamb, Omega):
    print("         grad_cost_U")
    result = 2*(P_Omega(U @ V.T @ X - Y, Omega) @ X.T @ V + \
            lamb * (Ly.T @ U @ V.T @ Lx.T + Ly.T @ Ly @ U @ V.T + \
            U @ V.T @ Lx.T @ Lx + Ly @ U @ V.T @ Lx) @ V)
    return result

def grad_cost_V(U ,V ,X, Y, Lx, Ly, lamb, Omega):
    print("         grad_cost_V")
    result = 2*(X @ P_Omega(U @ V.T @ X - Y, Omega).T @ U + \
            lamb * (Lx @ V @ U.T @ Ly + V @ U.T @ Ly.T @ Ly + \
            Lx.T @ Lx @ V @ U.T + Lx.T @ V @ U.T @ Ly.T) @ U)
    return result

def regularized_cost(U, V, X, Y, Lx, Ly, lamb, Omega):
    print("         regularized_cost")
    # print(U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)
    # print(U.shape, V.T.shape)
    # print(P_Omega(U @ V.T @ X - Y, Omega).shape)
    result = np.linalg.norm(P_Omega(U @ V.T @ X - Y, Omega), ord='fro')**2 + \
            lamb * np.linalg.norm(Ly @ U @ V.T + U @ V.T @ Lx.T, ord='fro')**2

# Return maximum cost if X has any elements < 0
def indicator_positive(X):
    if(X[X<0].size > 0):
        return np.finfo(X.dtype).max
    else:
        return 0

# Project Matrix X to nonnegative by setting all negative elements to 0.
# step size parameter 's' ignored (required for use as prox argument in for acc_prox_grad_method)
def proj_nonnegative(X, s=None):
    X[X<0] = 0
    return X