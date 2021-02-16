import numpy as np
import scipy
import math_util
import alt_acc_prox_grad




def optimize_alt_pgd(U, V,
                    X, Y, Lx, Ly, Omega,
                    lamb,
                    tol=1e-10,
                    max_outer_iter=10,
                    max_inner_iter=10,
                    max_line_iter=100,
                    calculate_cost=True):


    print("Performing PDG on regularized nonnegative regression problem.")
    U, V, costs = alt_acc_prox_grad.alternating_pgd(
                    U, V, 
                    lambda U, V: regularized_cost(U, V.T, X, Y, Lx, Ly, lamb, Omega),
                    lambda U, V: grad_cost_U(U, V.T, X, Y, Lx, Ly, lamb, Omega),
                    lambda U, V: grad_cost_V(U, V.T, X, Y, Lx, Ly, lamb, Omega).T, # note transpose
                    math_util.indicate_positive,
                    math_util.proj_nonnegative,
                    tol,
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost)
    return U, V, costs


def P_Omega(X, Omega):
    # print("P_omega")
    X[Omega.toarray()==1] = 0
    return X


def grad_cost_U(U, V, X, Y, Lx, Ly, lamb, Omega):
    # print("grad_U")
    result = U @ (V.T @ Lx.T @ V)
    result = result + Ly @ U @ (V.T @ V)
    result = Ly.T @ result
    result = result + U @ (V.T @ Lx.T@ Lx @ V)
    result = result + Ly @ U @ (V.T @ Lx @ V)
    result = lamb * result
    result = 2* (result + P_Omega(U @ (V.T @ X) - Y, Omega) @ (X.T @ V))

    return result

def grad_cost_V(U ,V ,X, Y, Lx, Ly, lamb, Omega):
    # print("grad_V")
    result = Lx @ V @ (U.T @ Ly @ U)
    result = result + V @ (U.T @ Ly.T @ Ly @ U)
    result = result + Lx.T @ Lx @ V @ (U.T @ U)
    result = result + Lx.T @ V @ (U.T @ Ly.T @ U)
    result = lamb * result
    result = 2* (result + X @ (P_Omega(U @ (V.T @ X) - Y, Omega).T @ U))

    return result


def regularized_cost(U, V, X, Y, Lx, Ly, lamb, Omega):
    # print("regularized_cost", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)
    result = np.linalg.norm(P_Omega(U @ (V.T @ X) - Y, Omega), ord='fro')**2
    result = result + lamb * math_util.factorized_sum_frobenius_sq(Ly @ U, V.T, U, V.T @ Lx.T)

    return result





