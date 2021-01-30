import numpy as np

# ||YZ+WH||F^2 using trace
def factorized_sum_frobenius_sq(W, H, Y, Z):
    trace1 = np.trace(H @ H.T @ W.T @ W)
    trace2 = np.trace(Z @ Z.T @ Y.T @ Y)
    trace3 = np.trace(H @ Z.T @ Y.T @ W)
    return trace1 + trace2 + 2*trace3

# ||YZ-WH||F^2 using trace
def factorized_difference_frobenius_sq(W, H, Y, Z):
    trace1 = np.trace(H @ H.T @ W.T @ W)
    trace2 = np.trace(Z @ Z.T @ Y.T @ Y)
    trace3 = np.trace(H @ Z.T @ Y.T @ W)
    return trace1 + trace2 - 2*trace3

# grad_W( ||YZ-WH||F^2 )
def grad_factorized_difference_frobenius_sq(W, H, Y, Z):
    # print("grad_frobenius_squared", W.shape, H.shape, Y.shape, Z.shape)
    return 2 * (W @ (H @ H.T) - (Y @ (Z @ H.T)) )

# Project Matrix X to nonnegative by setting all negative elements to 0.
# step size parameter 's' ignored (required for use as prox argument in for acc_prox_grad_method)
def proj_nonnegative(X, s=None):
    X[X<0] = 0
    return X

# Return maximum cost if X has any elements < 0
def indicate_positive(X):
    if(X[X<0].size > 0):
        return np.finfo(X.dtype).max
    else:
        return 0
