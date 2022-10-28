# imports
from keras.models import Sequential
from keras.layers import Flatten, Dense

# model construct
model = Sequential()

# Input layer
model.add(Flatten(input_shape = (1000,1000)))

# Dense layers
model.add(Dense(1000, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()