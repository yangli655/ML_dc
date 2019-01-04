import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    X_ = np.vstack((np.ones((1, N)), X))
    I = np.eye(P + 1)
    I[0, 0] = 0
    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_, X_.T) + lmbda * I), X_), y.T)
    # end answer
    return w
