import numpy as np

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Liklihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    '''

    C, N = x.shape
    l = np.zeros((C, N))
    #TODO

    # begin answer
    sum_ = x.sum(axis=1)
    for i in range(C):
        l[i] = x[i] / sum_[i]

    # end answer

    return l