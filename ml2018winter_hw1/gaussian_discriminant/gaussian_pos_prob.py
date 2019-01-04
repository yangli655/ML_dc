import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    likelihood_p = np.zeros((N, K))
    for i in range(K):
        temp1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma.T[i])))
        temp2 = np.linalg.pinv(Sigma.T[i])
        for j in range(N):
            tmp = X[:,j] - Mu[:,i]
            tmp_T = tmp[:, np.newaxis]
            likelihood_p[j][i] = temp1 * np.exp(-1/2 * np.mat(tmp) * temp2 * tmp_T)
    
    temp = likelihood_p * Phi.T
    temp_sum = temp.sum(axis=1)
    
    for i in range(N):
        p[i] = temp[i] / temp_sum[i]
    # end answer
    
    return p
    