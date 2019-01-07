import numpy as np
import matplotlib.pyplot as plt
from pca import PCA
from PIL import Image

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    #img_r = (plt.imread(filename)).astype(np.float64)
    # YOUR CODE HERE
    # begin answer
    img_r = Image.open(filename).convert('L')
    img_r = np.array(img_r, dtype=np.float64)

    coord = np.where(((img_r > 0) & (img_r < 150)))
    coord = np.array(coord).T
    
    eigval, eigvec = PCA(coord)

    new_cood = np.matmul(eigvec.T, coord.T) + 50
    new_cood = np.array(new_cood, dtype=np.int64)

    x = tuple(new_cood)
    
    img = np.zeros((img_r.shape[0]*3, img_r.shape[1]))
    img[x] = 50
    return img
    # end answer