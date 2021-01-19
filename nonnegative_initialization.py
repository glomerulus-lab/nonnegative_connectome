import numpy as np
from typing import Callable
import alt_acc_prox_grad

# Steps 1-3 of Atif et al
# Expects Y, Z such that X = Y @ Z
# Y shape (nx * p)
# Z shape (p * ny)

# Returns W, H such the X = W @ Z
# W shape (nx * (2p-1))
# H shape ((2p-1) * ny)
def init_nonnegative_factors(Y, Z):
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
    '''
    def alternating_pgd(Y, Z, W, H, 
                    cost_function,
                    grad_cost_function,
                    indicator_function,
                    projection_function,
                    max_outer_iter = 10,
                    max_inner_iter = 10,
                    max_line_iter = 100,
                    calculate_cost = False
                    ):
    '''

def refined_init_nonnegative_factors(Y, Z, 
                                    max_outer_iter = 10,
                                    max_inner_iter = 10,
                                    max_line_iter = 100,
                                    calculate_cost = False
                                    ):
    W, H = init_nonnegative_factors(Y,Z)
    
    print("Starting nonnegative factor refinement with PGD")
    W, H, costs = alt_acc_prox_grad.alternating_pgd(
                    W, H, 
                    lambda W, H: frobenius_squared_trace(W, H, Y, Z),
                    lambda W, H: grad_frobenius_squared(W, H, Y, Z),
                    lambda W, H: grad_frobenius_squared(H.T, W.T, Z.T, Y.T).T, # note transpose
                    indicator_positive,
                    proj_nonnegative,
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost
                    )
    return W, H, costs
   


#Cost, gradient, inicator, projection functions for use with alt_acc_prox_grad.py

# ||YZ-WH||F^2 using trace
def frobenius_squared_trace(W, H, Y, Z):
    # print("frobenius_squared_trace", W.shape, H.shape, Y.shape, Z.shape)
    trace1 = np.trace(H @ H.T @ W.T @ W)
    trace2 = np.trace(Z @ Z.T @ Y.T @ Y)
    trace3 = np.trace(H @ Z.T @ Y.T @ W)
    return trace1 + trace2 - 2*trace3

# grad_W( ||YZ-WH||F^2 )
def grad_frobenius_squared(W, H, Y, Z):
    # print("grad_frobenius_squared", W.shape, H.shape, Y.shape, Z.shape)
    return 2 * (W @ (H @ H.T) - (Y @ (Z @ H.T)) )


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
