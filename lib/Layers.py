# General DNN Ops...
import numpy as np
import theano, theano.tensor as T
import cPickle




def MLP(input, W_hidden, b_hidden, W_visible, b_visible, flatten=None):
    """ MLP Layer implementation """
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

def Unpool2D(input, pool_size):
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


# Conv2D and TConv2D for ResNet
def conv(input, filter_shape, W, border_mode=(0,0),
                        subsample=(1, 1)):
    """ Conv2D layer for Resnet architecture """
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

def deconv(input, filter_shape, input_shape, W,
                subsample=(1, 1),
               border_mode=(0,0)):
    """ Deconv2D for ResNet architecture """

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

def leaky_relu(x, alpha=0.2):
    return ((1 + alpha) * x + (1 - alpha) * abs(x)) / 2.

def dropout(X, p=0.):
    rng = np.random.RandomState(1)
    if p > 0:
        retain_prob = 1 - p
        X *= rng.binomial(X.shape, p=retain_prob).astype('float32')
        X /= retain_prob
    return X



# FIX THIS
#------------
def ResBlock(input, *args):
    """ Residual Block v1 http://arxiv.org/abs/1603.05027 """
    # Conv2D [3x3] with 'half' padding by default
    # Input -> Conv -> BN + NL -> Conv -> BN + NL -> sum -> NL -> output
    conv_1 = batchnorm(Conv2D(input=input, *args))
    conv_2 = batchnorm(Conv2D(input=conv_1, *args))
    sum = input + conv_2
    return relu(sum)