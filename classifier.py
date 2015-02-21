from IPython.core.debugger import Tracer
from autoencoder import DenoisingAutoencoder
from optimize import stochastic_gradient_descent
from skimage.io import imread_collection
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import os


tracer = Tracer()

folders = os.listdir('data/train')
train = np.empty((0, 2))

for folder in folders:
    examples = [[folder, os.path.join('data/normalized', folder, example)] for
                example in os.listdir(os.path.join('data/normalized', folder))]
    train = np.concatenate((train, examples), axis=0)

np.random.seed(0)
np.random.shuffle(train)
train_y = train[:, 0]
train_x = train[:, 1]
classes = list(set(train_y))
classes.sort()
print("Reading train images...")
train_x = np.array(imread_collection(train_x, conserve_memory=False)) / 255
train_x = train_x.reshape((len(train_x), -1))

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
x = T.matrix('x')
da = DenoisingAutoencoder(np_rng=rng, theano_rng=theano_rng, input=x,
                          n_visible=106 * 106, n_hidden=1800)

stochastic_gradient_descent(da, 0.1, train_x, 100, 3)
