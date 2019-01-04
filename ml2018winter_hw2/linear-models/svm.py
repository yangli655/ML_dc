import numpy as np
from scipy import optimize

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    X_ = np.vstack((np.ones((1, N)), X))
    b = np.ones((1, N))

    cons = ({'type': 'ineq', 'fun': lambda w:  (y * np.matmul(w.T, X_) - b).reshape(N)})
    res = optimize.minimize(lambda w: 1 / 2 * np.matmul(w.T, w), w, constraints = cons, method = 'SLSQP')
    w = res.x.reshape((P + 1, 1))

    for x in np.nditer(np.abs(np.matmul(w.T, X_) * y - 1)): 
        if x < 1e-6:
            num = num + 1
    # end answer
    return w, num

