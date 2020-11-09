from __future__ import print_function

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from core import Core, GenomeHandler


K.set_image_data_format("channels_last")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))



genome_handler = GenomeHandler(max_conv_layers=6, 
                               max_dense_layers=2, # includes final dense layer
                               max_filters=256,
                               max_dense_nodes=1024,
                               input_shape=x_train.shape[1:],
                               n_classes=10)

core2 = Core(genome_handler)
model = core2.run(dataset=dataset,
                  num_generations=20,
                  pop_size=20,
                  epochs=5)
print(model.summary())
