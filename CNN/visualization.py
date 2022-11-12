# import the dependencies
from sklearn.datasets import make_blobs
#The scikit-learb library to generate sample data
from keras.utils import to_categorical
#Keras methods that converts a class vector to a binary class matrix (one-hot encoding)

from keras import callbacks
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from matplotlib import pyplot
# Visualization library

# Generate a toy dataset
x, y = make_blobs(n_samples=1000, centers=3, n_features=2,
cluster_std=2, random_state=2)

# one-hot encode the label
y = to_categorical(y)

# split the dataset into 80% training data and 20% test data
n_train = 800
train_x, test_x = x[:n_train, :], x[n_train:, :]
train_y, test_y = y[:n_train], y[n_train:]
print(train_x.shape, test_x.shape)

# model
model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, padding='same',
#     activation='relu', input_shape=(32, 32, 3)))
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])
model.summary()

# Train the model for 1000 epochs
callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
    epochs=10000, verbose=1)

# Evaluate the model
_, train_acc = model.evaluate(train_x, train_y)
_, test_acc = model.evaluate(test_x, test_y)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc)) 

# plot the learning curves of model accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
