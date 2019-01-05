import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    N = X.shape[0]
    dis = np.zeros((N, N))
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dis[i, j] = dis[j, i] = np.sqrt(np.sum((X[i] - X[j])**2))
    dis = np.where(dis > threshold, np.inf, dis)

    for idx, item in enumerate(dis):
        idx_array = np.argsort(item)
        W[idx, idx_array[1:k+1]] = 1
    
    W = (W + W.T) / 2
    return W
    # end answer
