from scipy import fftpack
from skimage.transform import resize
from sklearn.base import TransformerMixin
import math
import numpy as np


class IdentityTransformer(TransformerMixin):

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x, **kwargs):
        return x

class NormalizeImages(TransformerMixin):

    def __init__(self, capture_percentage=0.8):
        self.capture_percentage = capture_percentage
        self.max_width = 0
        self.max_height = 0

    def fit(self, x, y, **kwargs):
        sizes = []
        for image in x:
            sizes.append(max(image.shape))
        sizes = np.array(sizes)

        capture_count = len(x) * self.capture_percentage

        for i in range(max(sizes)):
            self.size = i
            if (sizes < i).sum() >= capture_count:
                break

        return self

    def transform(self, x):
        output = np.full((len(x), self.size, self.size), 0)
        for i, image in enumerate(x):
            (image_height, image_width) = image.shape
            diff_y = math.floor(float(self.size - image_height) / 2.0)
            diff_x = math.floor(float(self.size - image_width) / 2.0)
            height = min(self.size, image_height)
            width = min(self.size, image_width)

            if diff_y >= 0:
                output_offset_y = diff_y
                image_offset_y = 0
            else:
                output_offset_y = 0
                image_offset_y = -diff_y

            if diff_x >= 0:
                output_offset_x = diff_x
                image_offset_x = 0
            else:
                output_offset_x = 0
                image_offset_x = -diff_x

            crop = image[
                image_offset_y:(image_offset_y+height),
                image_offset_x:(image_offset_x+width)
            ]
            output[
                i,
                output_offset_y:(output_offset_y+height),
                output_offset_x:(output_offset_x+width)
            ] = 255.0 - crop
        return output.reshape((len(x), -1)) / 255


class ResampleImages(TransformerMixin):

    def __init__(self, height, width=None):
        self.height = height
        self.width = width or height

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x, **kwargs):
        n, m = x.shape
        size = np.sqrt(m)
        output = np.empty((len(x), self.height, self.width))
        for i, image in enumerate(x):
            image = image.reshape((size, size))
            output[i] = resize(image, (self.height, self.width))
        return output.reshape((len(x), -1))


class FftTransformer(TransformerMixin):

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x, **kwargs):
        n, m = x.shape
        size = np.sqrt(m)
        output = np.empty((len(x), size, size))
        for i, image in enumerate(x):
            image = image.reshape((size, size))
            output[i] = np.abs(fftpack.fftshift(fftpack.fft2(image)))
        return output.reshape((len(x), -1))
