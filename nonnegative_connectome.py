import numpy as np
from typing import Callable
import alt_acc_prox_grad
import load_mat
import scipy


def optimize_alt_pgd(U, V,
                    X, Y, Lx, Ly, Omega,
                    lamb, 
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost=True):


    print("Performing PDG on regularized nonnegative regression problem.")
    U, V, costs = alt_acc_prox_grad.alternating_pgd(
                    U, V, 
                    lambda U, V: regularized_cost(U, V.T, X, Y, Lx, Ly, lamb, Omega),
                    lambda U, V: grad_cost_U(U, V.T, X, Y, Lx, Ly, lamb, Omega),
                    lambda U, V: grad_cost_V(U, V.T, X, Y, Lx, Ly, lamb, Omega).T, # note transpose
                    indicator_positive,
                    proj_nonnegative,
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
    result = result + lamb * frobenius_squared_trace(Ly @ U, V.T, U, V.T @ Lx.T)

    return result

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


# ||YZ+WH||F^2 using trace
def frobenius_squared_trace(W, H, Y, Z):
    # print("frobenius_squared_trace", W.shape, H.shape, Y.shape, Z.shape)
    trace1 = np.trace(H @ H.T @ W.T @ W)
    trace2 = np.trace(Z @ Z.T @ Y.T @ Y)
    trace3 = np.trace(H @ Z.T @ Y.T @ W)
    return trace1 + trace2 + 2*trace3

