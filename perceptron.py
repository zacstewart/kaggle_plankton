from logistic_regression import LogisticRegression
from theano import config
from theano import tensor as T
import numpy as np
import theano


class HiddenLayer(object):

    def __init__(self, np_rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        if W is None:
            W_values = np.asarray(
                np_rng.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high=np.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out)),
                dtype=config.floatX)
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        linear_output = T.dot(input, W) + b

        self.input = input
        self.W = W
        self.b = b

        if activation is None:
            self.output = linear_output
        else:
            self.output = activation(linear_output)

        self.params = [self.W, self.b]


class MultilayerPerceptron(object):

    def __init__(self, np_rng, input, n_in, n_hidden, n_out,
                 activation=T.tanh):
        self.hidden_layer = HiddenLayer(
            np_rng=np_rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation)

        self.logistic_regression_layer = LogisticRegression(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out)

        self.L1 = (abs(self.hidden_layer.W).sum() +
                   abs(self.logistic_regression_layer.W).sum())

        self.L2_sqr = ((self.hidden_layer.W ** 2).sum() +
                       (self.logistic_regression_layer.W ** 2).sum())

        self.p_of_y_given_x = self.logistic_regression_layer.p_of_y_given_x

        self.negative_log_likelihood = \
            self.logistic_regression_layer.negative_log_likelihood

        self.errors = self.logistic_regression_layer.errors

        self.params = \
            self.hidden_layer.params + self.logistic_regression_layer.params

    def cost_updates(self, y, learning_rate, L1_reg=0.0, L2_reg=0.0001):
        """Should L1_reg and L2_reg be attributes of the model?"""
        cost = (self.negative_log_likelihood(y) +
                L1_reg * self.L1 +
                L2_reg * self.L2_sqr)

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
