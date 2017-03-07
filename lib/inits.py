import numpy as np
import theano, theano.tensor as T

def He(shape, name, fan_in):
    """ He initialization of parameters """
    rng = np.random.RandomState(1)
    W = rng.normal(0, np.sqrt(2. / fan_in), size=shape)
    return theano.shared(W, borrow=True, name=name).astype('float32')

def ConstantInit(shape, name, value):
    """ Constant initialization of parameters """
    if value == 0:
        b = np.zeros(shape, dtype='float32')
        return theano.shared(b, name=name, borrow=True)