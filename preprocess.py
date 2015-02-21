from skimage.io import imread_collection, imsave
from transformers import NormalizeImages
import numpy as np
import os
from IPython.core.debugger import Tracer


tracer = Tracer()

folders = os.listdir('data/train')
train = np.empty((0, 3))

for folder in folders:
    examples = [[folder, os.path.join('data/train', folder, example), example] for
                example in os.listdir(os.path.join('data/train', folder))]
    train = np.concatenate((train, examples), axis=0)

train_y = train[:, 0]
train_x = train[:, 1]
filenames = train[:, 2]
train_x = np.array(imread_collection(train_x, conserve_memory=False)) / 255

tracer()

normalizer = NormalizeImages()

train_x_normalized = normalizer.fit_transform(train_x, train_y)
train_x_normalized = train_x_normalized.reshape(
    (len(train_x_normalized), normalizer.size, normalizer.size))

for class_, filename, image in zip(train_y, filenames, train_x_normalized):
    os.makedirs(os.path.join('data/normalized', class_), exist_ok=True)
    imsave(os.path.join('data/normalized', class_, filename), image)

tracer()
