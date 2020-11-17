#!/usr/bin/env python3

# %%

from Okeke_10152084_Train import model, batch_size
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
import tensorflow as tf
import random
import numpy as np
import os
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


warnings.filterwarnings('ignore')

# %%

# All images will be rescaled by 1./255
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
test_generator = test_datagen.flow_from_directory(
    'horse-or-human/validation/',  # This is the directory for testing images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# %%

test_predictions = model.predict(test_generator)
print('First 10 Predictions:')
print(test_predictions[:10])

(test_loss, test_acc, test_mse, test_bse,
 test_precision, test_recall, test_f1_score) = model.evaluate(
    test_generator)
print(f'Test Loss:\t{test_loss}')
print(f'Test Accuracy:\t{test_acc}')
print(f'Test MSE:\t{test_mse}')
print(f'Test Binary Crossentropy:\t{test_bse}')
print(f'Test Precision:\t{test_precision}')
print(f'Test Recall:\t{test_recall}')
print(f'Test F1-Score:\t{test_f1_score}')
