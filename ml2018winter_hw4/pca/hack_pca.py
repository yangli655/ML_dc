import numpy as np
from scipy import misc
from pca import PCA
from PIL import Image
from math import acos

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    # YOUR CODE HERE
    # begin answer
    img_r = Image.open(filename).convert('L')
    img_r = np.array(img_r, dtype=np.float64)

    coord = np.where(((img_r > 0) & (img_r < 100)))
    coord = np.array(coord).T
    
    eigval, eigvec = PCA(coord)
    angle = acos(eigvec[0,0] / np.linalg.norm(eigvec[:,0])) / np.pi * 180 - 90

    return misc.imrotate(img_r, angle)
    # end answer