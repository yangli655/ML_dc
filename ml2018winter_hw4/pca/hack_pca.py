import numpy as np
import matplotlib.pyplot as plt
from pca import PCA

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer
    #eigval, eigvec = PCA()
    return img_r
    # end answer