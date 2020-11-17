#!/usr/bin/env python3

# %%

import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import warnings
import tensorflow as tf
import random
import numpy as np
import os
from zipfile import ZipFile

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
tf.set_random_seed(42)


# %%

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# %%


# loading the data
local_zip = 'horse-or-human.zip'
zip_ref = ZipFile(local_zip, 'r')
zip_ref.extractall('horse-or-human')
zip_ref.close()

# %%

layers = [
    # Note the input shape is the desired size of the image
    # with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l1_l2(
                              l1=1e-5, l2=1e-4),
                          bias_regularizer=tf.keras.regularizers.l2(1e-4),
                          activity_regularizer=tf.keras.regularizers.l2(1e-5)
                          ),
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron. It will contain a value from 0-1
    # where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
]

batch_size = 64

model = Sequential(layers=layers)


def scheduler(epoch, lr):
    if epoch < 7:
        return lr
    else:
        return lr * math.exp(-0.1)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy',
                                     patience=3,
                                     restore_best_weights=True),
    tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1),
]

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              metrics=['acc', 'mean_squared_error', 'binary_crossentropy',
                       precision_m, recall_m, f1_score]
              )

# %%

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'horse-or-human/train/',  # This is the directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary',
    subset='training',
    seed=42
)

# Flow validation images in batches using train_datagen generator
validation_generator = train_datagen.flow_from_directory(
    'horse-or-human/train/',  # This is same directory
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary',
    subset='validation',
    seed=42
)

# %%

history = model.fit(
    train_generator,
    # number of samples / batch size
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)
