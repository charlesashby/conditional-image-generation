# General NN Ops...
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor.signal.pool as T_pool

def BinaryCrossEntropy(y_pred, y_true):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

def MeanSquaredError(y_pred, y_true):
    return T.sqr(y_pred - y_true).mean()

def wasserstein(y_pred, y_true):
    return T.mean(y_true * y_pred)

def concat(tensor_list, axis=1):
    return T.concatenate(tensor_list, axis)

def AbsoluteValue(y_pred, y_true):
    #return T.abs_(y_pred - y_true)
    return np.abs(y_pred - y_true)

# Taken from  https://github.com/Lasagne/Lasagne/blob/master/lasagne/nonlinearities.py
def elu(x):
    """Exponential Linear Unit :math:\\varphi(x) = (x > 0) ? x : e^x - 1
    The Exponential Linear Unit (ELU) was introduced in [1]_. Compared to the
    linear rectifier :func:rectify, it has a mean activation closer to zero
    and nonzero gradient for negative input, which can help convergence.
    Compared to the leaky rectifier :class:LeakyRectify, it saturates for
    highly negative inputs.
    Parameters
    ----------
    x : float32
        The activation (the summed, weighed input of a neuron).
    Returns
    -------
    float32
        The output of the exponential linear unit for the activation.
    Notes
    -----
    In [1]_, an additional parameter :math:\\alpha controls the (negative)
    saturation value for negative inputs, but is set to 1 for all experiments.
    It is omitted here.
    References
    ----------
    .. [1] Djork-Arne Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):
       Fast and Accurate Deep Network Learning by Exponential Linear Units
       (ELUs), http://arxiv.org/abs/1511.07289
    """
    return T.switch(x > 0, x, T.expm1(x))


def unpool(X, us):
    """ Unpooling layer
    X - input (N-D theano tensor of input images)
    us (tuple of length >= 1) - Factor by which to upscale (vertical ws, horizontal ws). (2,2) will double the image in each dimension. - Factors are applied to last dimensions of X
    """
    x_dims = X.ndim
    output = X
    for i, factor in enumerate(us[::-1]):
        if factor > 1:
            output = T.repeat(output, factor, x_dims - i - 1)
    return output

def resize_nearest_neighbour(inp, scale):
    inp_shp = T.shape(inp)
    upsample = T.tile(inp.dimshuffle(0,1,2,3,'x','x'),
        (scale * 2, scale * 2)).transpose(
            0,1,2,4,3,5).reshape((inp_shp[0], inp_shp[1],
                inp_shp[2]*scale*2, inp_shp[3]*scale*2))
    nn = T.nnet.neighbours.images2neibs(upsample,
                neib_shape=(scale, scale))

    interpolation = T.mean(nn, axis=1).dimshuffle(0, 'x')

    img = T.nnet.neighbours.neibs2images(interpolation,
                    neib_shape=(scale + 1, scale + 1),
                    original_shape=(inp_shp[0], inp_shp[1],
                    inp_shp[2]*scale / 2, inp_shp[3]*scale / 2))
    return img


def MLP(input, W_hidden, b_hidden, W_visible, b_visible, flatten=None):
    """ MLP Layer implementation """
    # TODO: add tied weight matrix
    if flatten == None:
        input = input
    else:
        input = input.flatten(2)
    H = T.nnet.relu(T.dot(input, W_hidden) + b_hidden)
    return T.nnet.sigmoid(T.dot(H, W_visible) + b_visible)

def Conv2D(input, filter_shape, input_shape,
           border_mode, subsample, W, b, reshape=None):
    """ Conv2D Layer Implementation """
    conv_out = T.nnet.conv2d(
        input=input,
        filters=W,
        filter_shape=filter_shape,
        input_shape=input_shape,
        border_mode=border_mode,
        subsample=subsample
    )

    return T.nnet.relu(conv_out + b.dimshuffle('x', 0, 'x', 'x'))


def pool2D(input, pool_size=(2, 2), mode='max'):
    """ Pooling Layer Implementation 2D """
    pooled_out = T.signal.pool.pool_2d(
        input=input,
        ws=pool_size,
        mode=mode,
        ignore_border=True
    )
    return pooled_out

def Unpool2D(input, pool_size=(2,2)):
    """ Unpooling Layer """

    # In this implementation of unpooling we simply repeat
    # the "maximal value" on the area that was previously
    # pooled.
    return input.repeat(pool_size[0], axis=2).\
        repeat(pool_size[1], axis=3)


def TransposedConv2D(input, filter_shape, input_shape,
                     subsample, W, b, border_mode=(0, 0), reshape=None):
    """ Transposed Convolution Implementation """
    if reshape == None:
        input = input
    else:
        input = input.reshape(reshape)

    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad=input,
        filters=W,
        input_shape=input_shape,
        filter_shape=filter_shape,
        border_mode=border_mode,
        subsample=subsample,
        filter_flip=True
    )

    return T.nnet.relu(deconv_out + b.dimshuffle('x', 0, 'x', 'x'))


# https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py
def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            _u = u.dimshuffle('x', 0, 'x', 'x')
            _s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            _u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            _s = T.mean(T.sqr(X - _u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            _u = (1. - a) * 0. + a * _u
            _s = (1. - a) * 1. + a * _s
        X = (X - _u) / T.sqrt(_s + e)
        if g is not None and b is not None:
            X = X * g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a) * 0. + a * u
            s = (1. - a) * 1. + a * s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X * g + b
    else:
        raise NotImplementedError
    return X

def convSqu(X, w, b=None, border_mode='half', subsample=(1, 1), filter_flip=True):

    output =\
        T.nnet.conv2d(
            input=X,
            filters=w,
            border_mode=border_mode,
            subsample=subsample,
            filter_flip=filter_flip)
    if b is not None:
        output += b.dimshuffle('x', 0, 'x', 'x')
    return output


# Conv2D and TConv2D for ResNet
def conv(input, filter_shape, W, border_mode=(0,0),
                        subsample=(1, 1)):
    """ Conv2D Layer Implementation """
    conv_out = T.nnet.conv2d(
        input=input,
        filters=W,
        filter_shape=filter_shape,
        border_mode=border_mode,
        subsample=subsample,
        filter_flip=True
    )

    return conv_out

def deconv(input, filter_shape,  W, input_shape,
                subsample=(1, 1),
               border_mode=(0,0)):
    """ Deconv2D """

    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad=input,
        filters=W,
        filter_shape=filter_shape,
        input_shape=input_shape,
        border_mode=border_mode,
        subsample=subsample,
        filter_flip=True
    )
    return deconv_out



# https://github.com/ppaquette/ift-6266-project/blob/master/lib/layers.py
def relu(x):
    return (x + abs(x)) / 2.

def lrelu(x, alpha=0.2):
    return ((1 + alpha) * x + (1 - alpha) * abs(x)) / 2.


def dropout(X, in_training, p=0.5):
    t_rng = RandomStreams(1)

    if p > 0:
        retain_prob = 1 - p
        X *= t_rng.binomial(X.shape, p=retain_prob).astype('float32')
        if not in_training:
            X /= retain_prob
    return X


def pool(X, ws, ignore_border=None, stride=None, pad=(0,0), mode='max'):

    if X.ndim >= 2 and len(ws) == 2:
        return T_pool.pool_2d(input=X, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode=mode)
    else:
        raise NotImplementedError

def max_pool(X, ws=(2,2), ignore_border=True, stride=None, pad=(0,0)):
    """ Max pooling layer - see pool() for details """
    return pool(X, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode='max')

def avg_pool(X, ws=(2,2), ignore_border=True, stride=None, pad=(0,0)):
    """ Average pooling layer - see pool() for details """
    return pool(X, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode='average_inc_pad')