import numpy as np
from typing import Callable
import alt_acc_prox_grad
import load_mat
import scipy


def optimize_alt_pgd(U, V,
                    testname,
                    lamb, 
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost=True):

    data = load_mat.load_all_matricies_data(testname)
    data["Omega"] = data["Omega"].astype(np.int8)
    data["Lx"] = data["Lx"].astype(np.int8)
    data["Ly"] = data["Ly"].astype(np.int8)
    print("Performing PDG on regularized nonnegative regression problem.")
    U, V, costs = alt_acc_prox_grad.alternating_pgd(
                    U, V, 
                    lambda U, V: regularized_cost(U, V.T, data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]),
                    lambda U, V: grad_cost_U(U, V.T, data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]),
                    lambda U, V: grad_cost_V(U, V.T, data["X"], data["Y"], data["Lx"], data["Ly"], lamb, data["Omega"]).T, # note transpose
                    indicator_positive,
                    proj_nonnegative,
                    max_outer_iter,
                    max_inner_iter,
                    max_line_iter,
                    calculate_cost)
    return U, V, costs


def P_Omega(X, Omega):
    #print("         P_Omega")
    X[Omega.toarray()==1] = 0
    #print("         P_Omega Done")
    return X




def grad_cost_U(U, V, X, Y, Lx, Ly, lamb, Omega):
    #print("         grad_cost_U", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)

    result = U @ (V.T @ Lx.T @ V)
    result = result + Ly @ U @ (V.T @ V)
    result = Ly.T @ result
    result = result + U @ (V.T @ Lx.T@ Lx @ V)
    result = result + Ly @ U @ (V.T @ Lx @ V)
    result = lamb * result
    result = 2* (result + P_Omega(U @ (V.T @ X) - Y, Omega) @ (X.T @ V))

    # result = np.linalg.multi_dot([Ly.T.toarray(), U, V.T, Lx.T.toarray(), V])
    # result = np.add(result, np.linalg.multi_dot([Ly.T.toarray(), Ly.toarray(), U, V.T, V]))
    # result = np.add(result, np.linalg.multi_dot([U, V.T, Lx.T.toarray(), Lx.toarray(), V]))
    # result = np.add(result, np.linalg.multi_dot([Ly.toarray(), U, V.T, Lx.toarray(), V]))
    # result = lamb * result
    # result = 2* (np.add(result, np.linalg.multi_dot([P_Omega(np.linalg.multi_dot([U, V.T, X]) - Y, Omega), X.T, V])))


    # result2 = 2*(P_Omega(U @ (V.T @ X) - Y, Omega) @ X.T @ V + \
    #         lamb * (Ly.T @ U @ V.T @ Lx.T + Ly.T @ Ly @ U @ V.T + \
    #         U @ V.T @ Lx.T @ Lx + Ly @ U @ V.T @ Lx) @ V)
    # print("grad_cost_U", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape, np.allclose(result,result2))
    # print("done")
    # exit()
    return result

def grad_cost_V(U ,V ,X, Y, Lx, Ly, lamb, Omega):
    # print("         grad_cost_V", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)

    result = Lx @ V @ (U.T @ Ly @ U)
    result = result + V @ (U.T @ Ly.T @ Ly @ U)
    result = result + Lx.T @ Lx @ V @ (U.T @ U)
    result = result + Lx.T @ V @ (U.T @ Ly.T @ U)
    result = lamb * result
    result = 2* (result + X @ (P_Omega(U @ (V.T @ X) - Y, Omega).T @ U))


    # result = np.linalg.multi_dot([Lx.toarray(), V, U.T, Ly.toarray(), U])
    # result = np.add(result, np.linalg.multi_dot([V, U.T, Ly.T.toarray(), Ly.toarray(), U]))
    # result = np.add(result, np.linalg.multi_dot([Lx.T.toarray(), Lx.toarray(), V, U.T, U]))
    # result = np.add(result, np.linalg.multi_dot([Lx.T.toarray(), V, U.T, Ly.T.toarray(), U]))
    # result = lamb * result
    # result = 2* (np.add(result, np.linalg.multi_dot([X, P_Omega(np.linalg.multi_dot([U, V.T, X]) - Y, Omega).T, U])))

    # result2 = 2*(X @ P_Omega(U @ V.T @ X - Y, Omega).T @ U + \
    #         lamb * (Lx @ V @ U.T @ Ly + V @ U.T @ Ly.T @ Ly + \
    #         Lx.T @ Lx @ V @ U.T + Lx.T @ V @ U.T @ Ly.T) @ U)
    # print("         grad_cost_V", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape, np.allclose(result,result2))
    # exit()

    return result

def regularized_cost(U, V, X, Y, Lx, Ly, lamb, Omega):
    # print("         regularized_cost", U.shape, V.shape, X.shape, Y.shape, Lx.shape, Ly.shape, Omega.shape)

    # result2 = np.linalg.norm(P_Omega(U @ V.T @ X - Y, Omega), ord='fro')**2 + \
    #         lamb * np.linalg.norm(Ly @ U @ V.T + U @ V.T @ Lx.T, ord='fro')**2

    result = np.linalg.norm(P_Omega(U @ (V.T @ X) - Y, Omega), ord='fro')**2
    result = result + lamb * frobenius_squared_trace(Ly @ U, V.T, U, V.T @ Lx.T)
    #print(np.allclose(result, result2))
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




'''
%% original formulation
function [gU]=grad_U_1(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gU=2*(-Y*X'*V + U*V'*X*X'*V +...
              lambda*(Ly'*U*V'*Lx' + Ly'*Ly*U*V' + ...
                      U*V'*Lx'*Lx + Ly*U*V'*Lx)*V);
    else
        gU=2*(P_Omega(U*V'*X-Y,Omega)*X'*V +...
              lambda*(Ly'*U*V'*Lx' + Ly'*Ly*U*V' + ...
                      U*V'*Lx'*Lx + Ly*U*V'*Lx)*V);
    end        
end

function [gV]=grad_V_1(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gV=2*(-X*Y'*U + X*X'*V*U'*U +...
              lambda*(Lx*V*U'*Ly + V*U'*Ly'*Ly +...
                      Lx'*Lx*V*U' + Lx'*V*U'*Ly')*U);
    else
        gV=2*(X*P_Omega(U*V'*X-Y,Omega)'*U + ...
              lambda*(Lx*V*U'*Ly + V*U'*Ly'*Ly +...
                      Lx'*Lx*V*U' + Lx'*V*U'*Ly')*U);
        
    end
end

function c=cost_1(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        c=norm(Y-U*V'*X,'fro')^2 + ...
          lambda*norm(U*V'*Lx' + Ly*U*V','fro')^2;
    else
        c=norm(P_Omega(Y-U*V'*X,Omega),'fro')^2 +...
          lambda*norm(U*V'*Lx' + Ly*U*V','fro')^2;
    end
end
'''