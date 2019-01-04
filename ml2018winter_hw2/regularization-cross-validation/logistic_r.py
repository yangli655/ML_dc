import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    X_ = np.vstack((np.ones((1, N)), X))
    y_ = np.where(y==1, 1, 0)
    cycles = 5000
    learning_rate = 0.2

    for i in range(cycles):
        temp = 1 / (1 + np.exp(-1 * np.matmul(w.T, X_))) - y_
        tmp = w
        tmp[0,0] = 0
        w = w - learning_rate * (np.matmul(X_, temp.T) + tmp) / N
    # end answer
    return w
