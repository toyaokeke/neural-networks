#!/usr/bin/env python3

# %%

from Okeke_10152084_Train import model
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
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
tf.random.set_seed(42)


warnings.filterwarnings('ignore')

# %%

# add file to folder when running this program
filename = input("""
                    Enter the file name and extension of the test data. 
                    This file must have EXACTLY 10,000 unique words for 
                    the model input dimensions as per the assignment guidelines:
                """)
sample = pd.read_excel(filename, index_col=0, header=None)
features = sample.iloc[:, [0]].values.flatten()
labels = sample.iloc[:, [1]].values.flatten()
vectorizer = CountVectorizer()
vectorized_data = vectorizer.fit_transform(features).toarray()

# %%

predictions = model.predict(vectorized_data)
print('Predictions:')
print(predictions)

loss, acc, mse = model.evaluate(vectorized_data, labels)
print(f'Test Loss:\t{loss}')
print(f'Test Accuracy:\t{acc}')
print(f'Test MSE:\t{mse}')

# %%
