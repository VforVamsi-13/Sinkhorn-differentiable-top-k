"""
citation : Rodrigo de Azevedo (https://math.stackexchange.com/users/339790/rodrigo-de-azevedo), Checking if a matrix has support, URL (version: 2025-11-03): https://math.stackexchange.com/q/5106720
"""


from cvxpy import *
import numpy as np

# determine whether square matrix A has support
def hasSupport(A):

    # extract the number of rows (or columns)
    n = A.shape[0]

    # extract the sparsity pattern of matrix A
    B = (A != 0).astype(int)
    
    # create column vector of n ones
    ones_n = np.ones((n,1))
    
    # create matrix decision variable
    X = Variable((n,n))

    # create linear program (LP)
    constraints = [            X @ ones_n == ones_n, 
                    ones_n.T @ X          == ones_n.T, 
                               X          >=      0   ]
    LP = Problem(Maximize(trace(B @ X)), constraints)
    LP.solve(verbose = False)
    
    # check if the optimization problem was solved
    if LP.status == "optimal":

        # maximal trace attained (rounded to integer)
        maxTrace = int(np.round(LP.value, 0))

        # return has-support-ness, maximal trace and permutation matrix
        return (maxTrace == n, maxTrace, np.round(X.value, 3))

    else:
        return "Error!"


def main():

    A = np.array([[ 1, 0, 0],
                  [ 0, 2, 3],
                  [ 0, 5, 0]])

    hasSupp, maxTrace, P = hasSupport(A)

    print("Matrix A has support:   ", hasSupp )
    print("Permutation matrix:     ", P       )
    print("Maximal trace:          ", maxTrace)
    print("A with permuted columns:", A @ P   )


if __name__ == "__main__":
    main()