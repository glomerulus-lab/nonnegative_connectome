#import numpy as np
# import scipy
import math_util
import alt_acc_prox_grad

import jax.numpy as np
from jax import jit
import jax


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
                    jit(lambda U, V: regularized_cost(U, V.T, X, Y, Lx, Ly, lamb, Omega)),
                    jit(lambda U, V: grad_cost_U(U, V.T, X, Y, Lx, Ly, lamb, Omega)),
                    jit(lambda U, V: grad_cost_V(U, V.T, X, Y, Lx, Ly, lamb, Omega).T), # note transpose
                    math_util.indicate_positive,
                    math_util.proj_nonnegative,
                    tol,
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost)
    return U, V, costs

# def P_Omega(X, Omega):
#     # print("P_omega")
#     # indicies = scipy.sparse.find(Omega)
#     # X[indicies[0], indicies[1]] = 0
#     return np.where(Omega==1, 0, X)

# def grad_cost_U(U, V, X, Y, Lx, Ly, lamb, Omega):
#     # print("grad_U")
#     result = U @ (V.T @ Lx.T @ V)
#     result = result + Ly @ U @ (V.T @ V)
#     result = Ly.T @ result
#     result = result + U @ (V.T @ Lx.T @ Lx @ V)
#     result = result + Ly @ U @ (V.T @ Lx @ V)
#     result = lamb * result
#     result = 2* (result + P_Omega(U @ (V.T @ X) - Y, Omega) @ (X.T @ V))

#     return result

# def grad_cost_V(U ,V ,X, Y, Lx, Ly, lamb, Omega):
#     # print("grad_V")
#     result = Lx @ V @ (U.T @ Ly @ U)
#     result = result + V @ (U.T @ Ly.T @ Ly @ U)
#     result = result + Lx.T @ Lx @ V @ (U.T @ U)
#     result = result + Lx.T @ V @ (U.T @ Ly.T @ U)
#     result = lamb * result
#     result = 2* (result + X @ (P_Omega(U @ (V.T @ X) - Y, Omega).T @ U))

#     return result

# def regularized_cost(U, V, X, Y, Lx, Ly, lamb, Omega):
#     # print("regularized_cost", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)
#     result = np.linalg.norm(P_Omega(U @ (V.T @ X) - Y, Omega), ord='fro')**2
#     result = result + lamb * math_util.factorized_difference_frobenius_sq(Ly @ U, V.T, U, V.T @ Lx.T)

#     return result


@jit
def P_Omega(X, Omega):
    # print("P_omega")
    # indicies = scipy.sparse.find(Omega)
    # X[indicies[0], indicies[1]] = 0
    return jax.ops.index_update(X, (Omega[0], Omega[1]), 0)

@jit
def grad_cost_U(U, V, X, Y, Lx, Ly, lamb, Omega):
    # print("grad_U")
    nx = X.shape[0]
    ny = Y.shape[0]
    result = U @ (sp_l_matmul(Lx, V, nx).T @ V)
    result = result + sp_l_matmul(Ly, U, ny) @ (V.T @ V)
    result = sp_l_matmul((Ly[1],Ly[0],Ly[2]), result, ny)
    result = result + U @ (sp_l_matmul(Lx, V, nx).T @ sp_l_matmul(Lx, V, nx))
    result = result + sp_l_matmul(Ly, U, ny) @ (V.T @ sp_l_matmul(Lx, V, nx))
    result = lamb * result
    result = 2* (result + P_Omega(U @ (V.T @ X) - Y, Omega) @ (X.T @ V))

    return result

@jit
def grad_cost_V(U ,V ,X, Y, Lx, Ly, lamb, Omega):
    # print("grad_V")
    nx = X.shape[0]
    ny = Y.shape[0]
    result = sp_l_matmul(Lx, V, nx) @ (U.T @ sp_l_matmul(Ly, U, ny))
    result = result + V @ (sp_l_matmul(Ly, U, ny).T @ sp_l_matmul(Ly, U, ny))
    result = result + sp_l_matmul((Lx[1],Lx[0],Lx[2]), sp_l_matmul(Lx, V, nx), nx) @ (U.T @ U)
    result = result + sp_l_matmul((Lx[1],Lx[0],Lx[2]), V, nx) @ (sp_l_matmul(Ly, U, ny).T @ U)
    result = lamb * result
    result = 2* (result + X @ (P_Omega(U @ (V.T @ X) - Y, Omega).T @ U))

    return result

@jit
def regularized_cost(U, V, X, Y, Lx, Ly, lamb, Omega):
    # print("regularized_cost", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)
    result = np.linalg.norm(P_Omega(U @ (V.T @ X) - Y, Omega), ord='fro')**2
    result = result + lamb * math_util.factorized_difference_frobenius_sq(sp_l_matmul(Ly, U, Y.shape[0]), V.T, U, sp_l_matmul(Lx, V, X.shape[0]).T)

    return result


# @jit
# def grad_cost_U(U, V, X, Y, Lx, Ly, lamb, Omega):
#     # print("grad_U")
#     nx = X.shape[0]
#     ny = Y.shape[0]
#     return 2* (lamb * (sp_l_matmul((Ly[1],Ly[0],Ly[2]), U @ (sp_l_matmul(Lx, V, nx).T @ V) + sp_l_matmul(Ly, U, ny) @ (V.T @ V), ny) + U @ (sp_l_matmul(Lx, V, nx).T @ sp_l_matmul(Lx, V, nx)) + sp_l_matmul(Ly, U, ny) @ (V.T @ sp_l_matmul(Lx, V, nx))) + P_Omega(U @ (V.T @ X) - Y, Omega) @ (X.T @ V))


# @jit
# def grad_cost_V(U ,V ,X, Y, Lx, Ly, lamb, Omega):
#     # print("grad_V")
#     nx = X.shape[0]
#     ny = Y.shape[0]
#     return 2* ((lamb * (sp_l_matmul(Lx, V, nx) @ (U.T @ sp_l_matmul(Ly, U, ny)) + V @ (sp_l_matmul(Ly, U, ny).T @ sp_l_matmul(Ly, U, ny)) + sp_l_matmul((Lx[1],Lx[0],Lx[2]), sp_l_matmul(Lx, V, nx), nx) @ (U.T @ U) + sp_l_matmul((Lx[1],Lx[0],Lx[2]), V, nx) @ (sp_l_matmul(Ly, U, ny).T @ U))) + X @ (P_Omega(U @ (V.T @ X) - Y, Omega).T @ U))




# http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/
@jax.partial(jax.jit, static_argnums=(2))
def sp_l_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (row indicies, col indicies, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    rows, cols, values = A
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res



