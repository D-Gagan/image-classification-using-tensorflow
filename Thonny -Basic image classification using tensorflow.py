import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)

from tensorflow.keras.datasets import mnist
(x_train, y_train).(x_test, y_test) = mnist.load.data()

print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)
print('x_test shape:',x_test.shape)
print('y_test shape:',y_test.shape)

from matplotlib import pyplot as plt
%matplotlib inline
plt.inshow(x_train[0],cmap='binary')
plt.show()

y_train[0]

print(set(y_train))

from tensorflow.keras.utils import to_categorial

y_train_encoded = to_categorial(y_train)
y_test_encoded = to_categorial(y_test)

print('y_train_encoded shape:',y_train_encoded.shape)
print('y_test_encoded shape:',y_test_encoded.shape)

y_train_encoded[0]

import numpy as np

x_train_reshaped = np.reshape(x_train,(60000,784))
x_test_reshaped = np.reshape(x_test,(10000,784))

print('x_train_reshaped:',x_train_reshaped.shape)
print('x_test_reshaped:',x_test_reshaped.shape)

print(set(x_train_reshaped[0]))

x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean)/(x_std + epsilon)
x_test_norm = (x_test_resshaped - x_mean)/(x_std + epsilon)

print(set(x_train_norm[0]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128 ,activation'relu',input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10,activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='categorial crossentrop',
    metrics=['accuracy']
)

model.summary()
