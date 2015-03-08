from skimage.io import imread_collection
import numpy as np


def chunk_generator(file_names, chunk_size):
    images = imread_collection(file_names)
    for idx in range(0, len(images), chunk_size):
        chunk_of_images = np.array(images[idx:idx + chunk_size])
        shape = chunk_of_images.shape
        chunk_of_images = chunk_of_images.reshape(
            shape[0], 1, shape[1], shape[2]) / 256
        yield chunk_of_images
