import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE
    # begin answer
    shuffle_x = x.copy()
    np.random.shuffle(shuffle_x)
    ctrs = shuffle_x[:k]
    iter_ctrs = []
    idx = np.zeros(x.shape[0], dtype=np.int64)
    changed = True
    while changed:
        changed = False
        s_distance = np.zeros((k, x.shape[0]))

        for i in range(k):
            temp = x - ctrs[i]
            s_distance[i] = np.sum(temp * temp, axis=1)
        t_idx = np.argmin(s_distance,axis=0)

        if (idx != t_idx).any():
            idx = t_idx
            changed = True
        
        for i in range(k):
            index = np.where(idx == i)
            ctrs[i] = np.mean(x[index],axis=0)
        iter_ctrs.append(ctrs)

    iter_ctrs = np.array(iter_ctrs)
    # end answer

    return idx, ctrs, iter_ctrs
