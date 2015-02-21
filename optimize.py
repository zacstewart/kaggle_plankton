from theano import tensor as T
import numpy as np
import theano


def stochastic_gradient_descent(
        da, learning_rate, train_set_x, batch_size, n_training_epochs):
    index = T.lscalar()
    train_set_x = theano.shared(value=train_set_x, name='train_set_x')

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    cost, updates = da.cost_updates(
        corruption_level=0.2, learning_rate=learning_rate)

    train = theano.function(
        [index], cost, updates=updates,
        givens={
            da.x: train_set_x[(index * batch_size):((index + 1) * batch_size)]
        })

    for epoch in range(n_training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train(batch_index))
        print("Training epoch {} cost {}".format(epoch, np.mean(c)))
