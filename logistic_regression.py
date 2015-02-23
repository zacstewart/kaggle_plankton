from theano import config
from theano import tensor as T
import numpy as np
import theano


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=config.floatX),
            name='W',
            borrow=True)
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=config.floatX),
            name='b',
            borrow=True)

        self.p_of_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_of_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_of_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        assert y.ndim == self.y_pred.ndim
        assert y.dtype.startswith('int')

        return T.mean(T.neq(self.y_pred, y))

    def cost_updates(self, y, learning_rate):
        cost = self.negative_log_likelihood(y)

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
