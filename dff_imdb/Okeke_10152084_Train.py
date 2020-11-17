#!/usr/bin/env python3

# %%

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import random
import numpy as np
import os

warnings.filterwarnings('ignore')


os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(42)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(42)


# %%


def plot_graphs(model):
    # Plot training & validation accuracy values
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation MSE
    plt.plot(model.history['mean_squared_error'])
    plt.plot(model.history['val_mean_squared_error'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # Plot training & validation loss values
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

# %%


def vectorize_sequences(sequences, dims=10000):
    results = np.zeros((len(sequences), dims))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# %%


(train_data, train_label), (test_data,
                            test_label) = tf.keras.datasets.imdb.load_data(num_words=10000, seed=42)
train_features = vectorize_sequences(train_data)
test_features = vectorize_sequences(test_data)

# %%

layers = [
    Dense(15, input_dim=10000, activation="relu"),
    Dropout(0.2),
    Dense(10, activation='sigmoid'),
    Dropout(0.2),
    Dense(5, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
]

model = Sequential(layers=layers)


def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=3,
                                     restore_best_weights=True),
    tf.keras.callbacks.LearningRateScheduler(schedule=scheduler),
]

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              metrics=['acc', 'mean_squared_error']
              )

# %%

history = model.fit(train_features, train_label,
                    epochs=20,
                    batch_size=256,
                    validation_split=0.2,
                    callbacks=callbacks,
                    )

# %%

plot_graphs(history)

# %%

test_predictions = model.predict(test_features)
print('Predictions:')
print(test_predictions)

test_loss, test_acc, test_mse = model.evaluate(test_features, test_label)
print(f'Test Loss:\t{test_loss}')
print(f'Test Accuracy:\t{test_acc}')
print(f'Test MSE:\t{test_mse}')

# %%
