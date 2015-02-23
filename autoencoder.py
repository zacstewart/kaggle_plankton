from theano import config
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano


class DenoisingAutoencoder(object):

    def __init__(self, np_rng, theano_rng=None, input=None, n_visible=11236,
                 n_hidden=1800, W=None, bvis=None, bhid=None):
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if not input:
            input = T.matrix(name='input')

        if not W:
            initial_W = np.asarray(np_rng.uniform(
                low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)), dtype=config.floatX)
            W = theano.shared(value=initial_W, name='W')

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(n_hidden, dtype=config.floatX), name='bhid')

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(n_visible, dtype=config.floatX), name='bvis')

        self.theano_rng = theano_rng
        self.x = input
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = W
        self.W_prime = self.W.T
        self.b = bhid
        self.b_prime = bvis
        self.params = [self.W, self.b, self.b_prime]

    def corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(
            size=input.shape, n=1, p=1 - corruption_level) * input

    def hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.corrupted_input(self.x, corruption_level)
        y = self.hidden_values(tilde_x)
        z = self.reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
