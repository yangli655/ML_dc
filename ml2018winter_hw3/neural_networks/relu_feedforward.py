import numpy as np

def relu_feedforward(in_):
    '''
    The feedward process of relu
      in_:
              in_	: the input, shape: any shape of matrix
      
      outputs:
              out : the output, shape: same as in
    '''
    # TODO

    # begin answer
    out = np.where(in_ >= 0, in_, 0)
    # end answer
    return out
