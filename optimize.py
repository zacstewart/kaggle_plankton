from theano import config
from theano import tensor as T
import numpy as np
import theano


def stochastic_gradient_descent(
        model, train_set_x, train_set_y, validate_set_x, validate_set_y, x, y,
        learning_rate, batch_size, n_training_epochs,
        patience_increase=2, improvement_threshold=0.995, **cost_kwargs):

    patience = 100000
    n_train_batches = train_set_x.shape[0] // batch_size
    n_validate_batches = validate_set_x.shape[0] // batch_size
    validation_frequency = min(n_train_batches, patience / 2)

    train_set_x = theano.shared(
        borrow=True,
        value=train_set_x.astype(config.floatX),
        name='train_set_x')
    train_set_y = theano.shared(
        borrow=True,
        value=train_set_y,
        name='train_set_y')
    validate_set_x = theano.shared(
        borrow=True,
        value=validate_set_x.astype(config.floatX),
        name='validate_set_x')
    validate_set_y = theano.shared(
        borrow=True,
        value=validate_set_y,
        name='validate_set_y')

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

    validate = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: validate_set_x[(index * batch_size):((index + 1) * batch_size)],
            y: validate_set_y[(index * batch_size):((index + 1) * batch_size)]
        })

    best_validation_loss = np.inf
    for epoch in range(n_training_epochs):
        costs = []
        for batch_index in range(n_train_batches):
            batch_cost = train(batch_index)
            costs.append(batch_cost)

            iteration = epoch * n_train_batches + batch_index
            if (iteration + 1) % validation_frequency == 0:
                batch_validation_loss = np.mean(
                    [validate(i) for i in range(n_validate_batches)])
                print("Epoch {}, batch {}/{}, validation error: {}".format(
                    epoch, batch_index, n_train_batches, batch_validation_loss
                ))

                if batch_validation_loss < best_validation_loss:
                    if (batch_validation_loss <
                            best_validation_loss * improvement_threshold):
                        patience = max(patience, iteration * patience_increase)
                    best_validation_loss = batch_validation_loss

            if iteration >= patience:
                return

        print("Epoch {} cost: {}".format(epoch, np.mean(costs)))
