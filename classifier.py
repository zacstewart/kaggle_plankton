from IPython.core.debugger import Tracer
from cnn import ConvolutionMaxPoolLayer
from perceptron import MultilayerPerceptron
from optimize import stochastic_gradient_descent
from pandas import DataFrame
from skimage.io import imread_collection
from theano import config
from theano import tensor as T
import data
import numpy as np
import os
import theano


tracer = Tracer()
TRAIN_DATA_ROOT = '.'
TRAIN_DATA_DIR = os.path.join(TRAIN_DATA_ROOT, 'data', 'train_normalized')

folders = os.listdir(TRAIN_DATA_DIR)
train = np.empty((0, 2))

for folder in folders:
    examples = [
        [folder, os.path.join(TRAIN_DATA_DIR, folder, example)] for
        example in os.listdir(os.path.join(TRAIN_DATA_DIR, folder))]
    train = np.concatenate((train, examples), axis=0)

np.random.seed(0)
np.random.shuffle(train)
validate = train[:(0.2 * len(train))]
train = train[(0.2 * len(train)):]
train_y = train[:, 0]
train_x = train[:, 1]
validate_y = validate[:, 0]
validate_x = validate[:, 1]
classes = list(set(train_y))
classes.sort()
i_to_class = dict(zip(range(len(classes)), classes))
class_to_i = {c: i for i, c in i_to_class.items()}
train_y = np.array(list(map(lambda y: class_to_i[y], train_y)), dtype=np.int32)
validate_y = np.array(list(map(lambda y: class_to_i[y], validate_y)), dtype=np.int32)

print("Reading train images...")

train_x = np.array(imread_collection(train_x, conserve_memory=False)) / 256
validate_x = np.array(imread_collection(validate_x, conserve_memory=False)) / 256
validate_x = validate_x.reshape((-1, 1, 106, 106))  # 1-color feature map
train_x = train_x.reshape((-1, 1, 106, 106))  # 1-color feature map

print("Building network symbolic graph...")

np_rng = np.random.RandomState(0)
x = T.tensor4('x', dtype=config.floatX)
y = T.ivector('y')
batch_size = 5

layer0 = ConvolutionMaxPoolLayer(
    np_rng=np_rng,
    input=x,
    image_shape=(batch_size, 1, 106, 106),
    filter_shape=(20, 1, 19, 19),
    pool_size=(2, 2))

layer1 = ConvolutionMaxPoolLayer(
    np_rng=np_rng,
    input=layer0.output,
    image_shape=(batch_size, 20, 44, 44),
    filter_shape=(50, 20, 11, 11),
    pool_size=(2, 2))

laye2_input = layer1.output.flatten(2)
layer2 = MultilayerPerceptron(
    np_rng=np_rng,
    input=laye2_input,
    n_in=50 * 17 * 17,
    n_hidden=1000,
    n_out=len(classes),
    activation=T.nnet.sigmoid)

print("Training network...")
stochastic_gradient_descent(
    layer2, train_x, train_y, validate_x, validate_y, x, y, learning_rate=0.1,
    batch_size=batch_size, n_training_epochs=1, L1_reg=0.0, L2_reg=0.0001)


print("Loading test set..")
images = np.array(os.listdir('data/test_normalized'))
test_file_names = np.array([os.path.join('data/test_normalized', filename)
                           for filename in os.listdir('data/test_normalized')])


print("Building network symbolic graph...")

chunk_size = 5000
test_set_x = theano.shared(
    value=np.zeros((1, 1, 1, 1), dtype=config.floatX),
    borrow=True,
    name='test_set_x')
index = T.lscalar()
predict_proba = theano.function(
    inputs=[index],
    outputs=layer2.p_of_y_given_x,
    givens={
        x: test_set_x[(index * batch_size):((index + 1) * batch_size)],
    })

print("Predicting...")

all_probabilities = []
for test_x_data in data.chunk_generator(test_file_names, chunk_size):
    this_chunk_size = min(chunk_size, test_x_data.shape[0])
    n_test_batches = int(np.ceil(this_chunk_size / batch_size))
    test_set_x.set_value(test_x_data.astype(config.floatX))
    for batch_index in range(n_test_batches):
        batch_probabilities = predict_proba(batch_index)
        all_probabilities.append(batch_probabilities)
all_probabilities = np.vstack(all_probabilities)

submission = DataFrame(
    columns=list(map(lambda i: i_to_class[i], range(len(classes)))),
    data=all_probabilities)
submission['image'] = images
submission.to_csv('submission.csv', index=False)

tracer()
