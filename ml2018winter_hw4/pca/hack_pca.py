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
    R, G, B, A = img_r[:,:,0], img_r[:,:,1], img_r[:,:,2], img_r[:,:,3]
    Red = ((1 - A) * R) + (A * R)
    Green = ((1 - A) * G) + (A * G)
    Blue = ((1 - A) * B) + (A * B)
    gray = 0.2989 * Red + 0.5870 * Green + 0.1140 * Blue

    eigval, eigvec = PCA(gray)

    gray_avg = np.average(gray, axis=0)
    d = gray - gray_avg
    r = np.matmul(d, eigvec) + gray_avg
    return r.astype(np.float64)
    # end answer