import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_minist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_minist.load_data()

