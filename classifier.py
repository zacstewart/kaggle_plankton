from pandas import DataFrame
from skimage.io import imread_collection
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from transformers import (
    IdentityTransformer, NormalizeImages, ResampleImages, FftTransformer)
import numpy as np
import os
from IPython.core.debugger import Tracer
tracer = Tracer()

folders = os.listdir('data/train')
train = np.empty((0, 2))

for folder in folders:
    examples = [[folder, os.path.join('data/train', folder, example)] for
                example in os.listdir(os.path.join('data/train', folder))]
    train = np.concatenate((train, examples), axis=0)

np.random.seed(0)
np.random.shuffle(train)
train_y = train[:, 0]
train_x = train[:, 1]
classes = list(set(train_y))
classes.sort()
print("Reading train images...")
train_x = np.array(imread_collection(train_x, conserve_memory=False))

kf = StratifiedKFold(train_y, n_folds=3, shuffle=True)

pipeline = Pipeline([
    ('normalize_imgs', NormalizeImages(capture_percentage=.8)),
    ('resized_imgs', ResampleImages(32)),
    ('features', FeatureUnion([
        ('image', IdentityTransformer()),
        ('fft', FftTransformer())
    ])),
    ('classifier', SVC(probability=True, verbose=True))
])

print("Cross validating...")
scores = []
for fold, (construct_idx, validate_idx) in enumerate(kf):
    print("Fold {}...".format(fold + 1))
    pipeline.fit(train_x[construct_idx], train_y[construct_idx])
    predictions = pipeline.predict_proba(train_x[validate_idx])
    score = metrics.log_loss(train_y[validate_idx], predictions)
    scores.append(score)

print("Mean score:", np.mean(scores))
print("Score std:", np.std(scores))
print("Raw scores:", scores)

print("Training...")

pipeline.fit(train_x, train_y)

print("Loading test images...")

images = np.array(os.listdir('data/test')).reshape((-1, 1))
test = np.array([os.path.join('data/test', filename)
                for filename in os.listdir('data/test')])
test_x = np.array(imread_collection(test, conserve_memory=False))

print("Predicting...")

probabilities = pipeline.predict_proba(test_x)
submission = DataFrame(data=probabilities, columns=classes)
submission['image'] = images
data = np.append(images, probabilities, axis=1)
submission.to_csv('submission.csv', index=False)

tracer()
