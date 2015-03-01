from theano import config
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
import theano


class ConvolutionMaxPoolLayer(object):

    def __init__(self, np_rng, input, image_shape, filter_shape,
                 pool_size=(2, 2)):
        assert filter_shape[1] == image_shape[1]

        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = \
            filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size)

        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        W_values = np.asarray(
            np_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        convoluted_output = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)

        pooled_out = downsample.max_pool_2d(
            input=convoluted_output,
            ds=pool_size,
            ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
