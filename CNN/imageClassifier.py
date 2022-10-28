from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Builds the model object
model = Sequential()

# CONV_1 adds a convolutional layer with ReLU activation and
# depth = 32 kernels
model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same',
activation='relu', input_shape=(28, 28, 1)))

# POOL_1 downsamples the image to choose the best features
model.add(MaxPooling2D(pool_size=(2, 2)))

# CONV_2 increases the depth of 64
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))

# POOL_2: more downsampling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten, since there are too many dimensions; we only want
# a classification output
model.add(Flatten())

# Dropout layer
model.add(Dropout(rate=0.3))

# FC_1: Fully connected to get all relevant data
model.add(Dense(64, activation='relu'))

# Dropout layer
model.add(Dropout(rate=0.5))

# FC_2: Outputs a softmax to squash the matrix into the
# 10 classes
model.add(Dense(10, activation='softmax'))

# prints the model architecture summary
model.summary()
