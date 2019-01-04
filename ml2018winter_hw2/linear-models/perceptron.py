import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    X_ = np.vstack((np.ones((1, N)), X))
    modify = True
    
    while modify:
        modify = False
        for i in range(N):
            x_t = X_[:,i].reshape(P + 1, 1)
            if np.vdot(w, x_t) * y[0,i] <= 0:
                w = w + y[0,i] * x_t
                iters = iters + 1
                modify = True
    # end answer
    
    return w, iters