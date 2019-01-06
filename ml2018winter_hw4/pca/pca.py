import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    data_avg = np.average(data, axis=0)
    #data_std = np.std(data, axis=0)
    #d = (data - data_avg) / data_std
    d = data - data_avg
    cov_data = np.cov(d.T)
    eigval, eigvec = np.linalg.eig(cov_data)

    sorted_eig_idx = np.argsort(-eigval)
    eigvalue = eigval[sorted_eig_idx]
    eigvector = eigvec[:, sorted_eig_idx]

    print(eigvector.shape)
    return eigvalue, eigvector
    # end answer