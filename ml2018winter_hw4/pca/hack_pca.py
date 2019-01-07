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
    print(img_r.shape[0] * img_r.shape[1])
    print(np.sum(img_r ==0))
    p = img_r.shape[1]
    eigval, eigvec = PCA(img_r)
    base = np.matmul(img_r, eigvec).astype(np.int64)
    
    return img_r[base]
    # end answer