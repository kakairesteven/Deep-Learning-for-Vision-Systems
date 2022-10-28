import numpy as np
from keras.models import Sequential
from keras.layers import Flatten

# Input layer
model = Sequential()
model.add(Flatten(input_shape = (28, 28)))

# print(model)

# Hidden layers
from keras.layers import Dense
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

# print(model)

# Output layers
model.add(Dense(10, activation='softmax'))


# CREATING A CONVOLUTIONAL LAYER
# Import keras
from keras.layers import Conv2D, MaxPooling2D

model.add(Conv2D(filters=16, kernel_size=2, strides='1',
padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.summary()