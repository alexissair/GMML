import numpy as np 

def FloydWarshall(A):
    '''A is the matrix of the graph :
            - A_ij = d(i, j) if i and j are connected
            - A_ij = np.inf otherwise'''
    n = A.shape[0]
    W = A.copy()
    # For computation reasons, we can't have np.inf in our geodesic path when we call MMDS
    for k in range(n) :
        for i in range(n) :
            for j in range(n) :
                W[i, j] = min(W[i,j], W[i, k] + W[k, j])
    return W

