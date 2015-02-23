from IPython.core.debugger import Tracer
from logistic_regression import LogisticRegression
from optimize import stochastic_gradient_descent
from pandas import DataFrame
from skimage.io import imread_collection
from theano import tensor as T
import numpy as np
import os
import theano


tracer = Tracer()

folders = os.listdir('data/train')
train = np.empty((0, 2))

for folder in folders:
    examples = [
        [folder, os.path.join('data/train_normalized', folder, example)] for
        example in os.listdir(os.path.join('data/train_normalized', folder))]
    train = np.concatenate((train, examples), axis=0)

np.random.seed(0)
np.random.shuffle(train)
train_y = train[:, 0]
train_x = train[:, 1]
classes = list(set(train_y))
classes.sort()
i_to_class = dict(zip(range(len(classes)), classes))
class_to_i = {c: i for i, c in i_to_class.items()}
train_y = np.array(list(map(lambda y: class_to_i[y], train_y)), dtype=np.int32)

print("Reading train images...")
train_x = np.array(imread_collection(train_x, conserve_memory=False)) / 255
train_x = train_x.reshape((len(train_x), -1))

x = T.matrix('x')
y = T.ivector('y')
lr = LogisticRegression(input=x, n_in=106 * 106, n_out=len(classes))
stochastic_gradient_descent(lr, train_x, train_y, x, y, 0.1, 100, 1)

predict_proba = theano.function([x], lr.p_of_y_given_x)

print("Loading test set..")
images = np.array(os.listdir('data/test_normalized'))
test = np.array([os.path.join('data/test_normalized', filename)
                for filename in os.listdir('data/test_normalized')])
test_set_x = np.array(imread_collection(test, conserve_memory=False)) / 255
test_set_x = test_set_x.reshape((len(test_set_x), -1))
tracer()
probabilities = predict_proba(test_set_x)
submission = DataFrame(
    data=probabilities,
    columns=list(map(lambda i: i_to_class[i], range(len(classes)))))
submission['image'] = images
submission.to_csv('submission.csv', index=False)

tracer()
