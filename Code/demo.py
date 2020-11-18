from __future__ import print_function

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
from keras import backend as K
from core import Core, GenomeHandler
import pandas as pd

K.set_image_data_format("channels_last")

query = input("Private or Public Dataset?")

if (query == "Private"):
        
    df = pd.read_csv(input("Enter file name "))

    X = df.values[:,1:]
    y = df.values[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #y_train, y_test = to_categorical(y_train, y_test)
    model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid'),])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=32, epochs=100)
    model.evaluate(X_test, y_test)[1]

else:
    print("Private Dataset not provided, using MNIST/CIFAR10/CIFAR100")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    dataset = ((x_train, y_train), (x_test, y_test))


    genome_handler = GenomeHandler(max_conv_layers=6, 
                                max_dense_layers=4,
                                max_filters=256,
                                max_dense_nodes=1024,
                                input_shape=x_train.shape[1:],
                                n_classes=10)

    core2 = Core(genome_handler)
    model = core2.run(dataset=dataset,
                    num_generations=1,
                    pop_size=1,
                    epochs=1)
    print(model.summary()) 
