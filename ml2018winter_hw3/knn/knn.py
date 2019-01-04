import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE
    # begin answer
    N_test = x.shape[0]
    y = np.zeros(N_test)

    for i in range(N_test):
        diff = x_train - x[i,:]
        square_dist = np.sum(diff * diff, axis=1)
        sorted_index = np.argsort(square_dist)
        nodes_label = []

        for j in range(k):
            index = sorted_index[j]
            nodes_label.append(y_train[index])

        y[i] = scipy.stats.mode(nodes_label).mode[0]
    # end answer

    return y
