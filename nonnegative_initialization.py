import numpy as np
import alt_acc_prox_grad
import math_util

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

def refine_nonnegative_factors(W, H, Y, Z,
                                tol=1e-7,
                                max_outer_iter = 10,
                                max_inner_iter = 10,
                                max_line_iter = 100,
                                calculate_cost = False
                                ):
   
    print("Starting nonnegative factor refinement with PGD")
    W, H, costs = alt_acc_prox_grad.alternating_pgd(
                    W, H, 
                    lambda W, H: math_util.factorized_difference_frobenius_sq(W, H, Y, Z),
                    lambda W, H: math_util.grad_factorized_difference_frobenius_sq(W, H, Y, Z),
                    lambda W, H: math_util.grad_factorized_difference_frobenius_sq(H.T, W.T, Z.T, Y.T).T, # note transpose of result
                    math_util.indicate_positive,
                    math_util.proj_nonnegative,
                    tol,
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost
                    )
    return W, H, costs