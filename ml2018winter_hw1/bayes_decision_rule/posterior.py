import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO

    # begin answer
    priori = np.zeros((C, 1))
    sum_ = x.sum(axis=1)

    for i in range(C):
        priori[i] = sum_[i] / total
    
    e = np.zeros(N)

    for i in range(N):
        for j in range(C):
            e[i] = e[i] + priori[j] * l[j,i]

    for i in range(N):
        for j in range(C):
            p[j,i] = l[j,i] * priori[j] / e[i]
            
    # end answer
    
    return p
