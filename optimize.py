from theano import config
from theano import tensor as T
import numpy as np
import theano


def stochastic_gradient_descent(
        model, train_set_x, train_set_y, x, y, learning_rate, batch_size,
        n_training_epochs, **cost_kwargs):

    n_train_batches = train_set_x.shape[0] // batch_size
    train_set_x = theano.shared(
        value=train_set_x.astype(config.floatX), name='train_set_x')
    train_set_y = theano.shared(
        value=train_set_y, name='train_set_y')

    index = T.lscalar()

    cost, updates = model.cost_updates(
        y, learning_rate=learning_rate, **cost_kwargs)

    train = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[(index * batch_size):((index + 1) * batch_size)],
            y: train_set_y[(index * batch_size):((index + 1) * batch_size)]
        })

    for epoch in range(n_training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            score = train(batch_index)
            print("Batch {} score {}".format(batch_index, score))
            c.append(score)
        print("Training epoch {} cost {}".format(epoch, np.mean(c)))
