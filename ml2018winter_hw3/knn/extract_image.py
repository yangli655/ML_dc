import numpy as np
import scipy.misc
def extract_image(image_file_name):
    '''
    EXTRACT_IMAGE Extract features from image
      Inputs:
          image_file_name: filename of image
      Outputs:
          x: 5x140 matrix, 5 digits in an image, each digit is a (140, 1) column vector.
    '''
    # m = imread(image_file_name)
    m = scipy.misc.imread(image_file_name, mode='L')
    d1 = m[4:18, 4:14].reshape(140)
    d2 = m[4:18, 13:23].reshape(140)
    d3 = m[4:18, 22:32].reshape(140)
    d4 = m[4:18, 31:41].reshape(140)
    d5 = m[4:18, 40:50].reshape(140)
    x = np.vstack((d1, d2, d3, d4, d5))
    return x
