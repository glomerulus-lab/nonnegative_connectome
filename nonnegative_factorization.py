import scipy.io
import numpy as np
from typing import Callable
import optimization



# Steps 1-3 of Atif et al
# Expects Y, Z such that X = Y @ Z
# Y shape (nx * p)
# Z shape (p * ny)

# Returns W, H such the X = W @ Z
# W shape (nx * (2p-1))
# H shape ((2p-1) * ny)
def initialize_nonnegative_factors(Y, Z):
    # rank of new nonnegative factors
    rank = (Y.shape[1])*2-1

    #Create empty arrays for W, H
    W = np.empty((Y.shape[0], rank))
    H = np.empty((rank, Z.shape[1]))

    #Initialize W and H with the absolute value of the first rank-one factors
    W[:,0] = np.abs(Y[:,0])
    H[0,:] = np.abs(Z[0,:])

    #For use in the upcoming loop, calculate Y(>=0), Y(<=0), Z(>=0), Z(<=0)
    Y_pos = np.where(Y < 0, 0, Y)
    Y_neg = np.where(Y < 0, abs(Y), 0)

    Z_pos = np.where(Z < 0, 0, Z)
    Z_neg = np.where(Z < 0, abs(Z), 0)

    #Build new nonnegative W, H

    j = 1
    for i in range(1, rank):
        if i % 2 != 0:
            W[:,i] = Y_pos[:,j]
            H[i,:] = Z_pos[j,:]
        else:
            W[:,i] = Y_neg[:,j]
            H[i,:] = Z_neg[j,:]
            j += 1

    return W, H



# cost_function(Y, Z, W ,H) for W
# grad_cost_function(Y, Z, W ,H) for W
def alternating_pgd(Y, Z, W, H, 
                    cost_function,
                    grad_cost_function,
                    indicator_function,
                    projection_function,
                    max_outer_iter = 100,
                    max_inner_iter = 100,
                    max_line_iter = 100,
                    ):
    W_k = W
    H_k = H
    k = 0

    costs = []
    while(k < max_outer_iter):
        # Get functions in terms of W
        g = alternating_cost_fun(cost_function, Y, Z, H_k)
        grad_g = alternating_cost_fun(grad_cost_function, Y, Z , H_k)

        W_k = optimization.acc_prox_grad_method(W_k, 
                                                g, 
                                                grad_g, 
                                                indicator_function, 
                                                projection_function, 
                                                max_iter=max_inner_iter, 
                                                max_line_iter=max_line_iter)

        if(indicator_function(W_k > 0)):
            print("indicator function W true")
            print(W_k[W_k<0])
            exit()
        # Get functions in terms of H_k.T
        g = alternating_cost_fun(cost_function, Z.T, Y.T, W_k.T)
        grad_g = alternating_cost_fun(grad_cost_function, Z.T, Y.T, W_k.T)
        
        H_k = optimization.acc_prox_grad_method(H_k.T, 
                                                g, 
                                                grad_g, 
                                                indicator_function, 
                                                projection_function, 
                                                max_iter=max_inner_iter, 
                                                max_line_iter=max_line_iter)
        H_k = H_k.T

        if(indicator_function(H_k > 0)):
            print("indicator function H true")
            print(H_k[H_k<0])
            exit()
        
        k+=1
        cost = cost_function(W_k, H_k, Y, Z)
        print(k, cost)
        costs.append(cost)

    return W_k, H_k, costs


def alternating_cost_fun(f, Y, Z, H):
    return lambda W: f(Y, Z, W, H)





def frobenius_squared_trace(Y, Z, W, H):
    trace1 = np.trace(H @ H.T @ W.T @ W)
    trace2 = np.trace(Z @ Z.T @ Y.T @ Y)
    trace3 = np.trace(H @ Z.T @ Y.T @ W)
    return trace1 + trace2 - 2*trace3


def grad_frobenius_squared(Y, Z, W, H):
    A = (H @ H.T)
    B = Y @ (Z @ H.T)
    return 2 * (W @ A - B)


# Return maximum cost if X has any negative values
def indicator_positive(X):
    if(X[X<0].size > 0):
        return np.finfo(X.dtype).max
    else:
        return 0


#step size parameter ignored
def proj_nonnegative(X, tmp=None):
    X[X<0] = 0
    return X