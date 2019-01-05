import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    N = W.shape[0]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eigval, eigvec = np.linalg.eig(L)
    eigval_idx = np.argsort(eigval)[0:k]
    eigvec_idx = eigvec[:, eigval_idx]
    idx = kmeans(eigvec_idx, k)
    return idx
    # end answer
