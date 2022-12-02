# Imports

import tensorflow as tf
from tensorflow import convert_to_tensor, string, matmul, math, cast, float32, reshape, shape, transpose, linalg, ones
from tensorflow import maximum, newaxis, convert_to_tensor, int64, data, train, reduce_sum, equal, argmax
from tensorflow import GradientTape, TensorSpec, function
from tensorflow.keras.layers import TextVectorization, Embedding, Layer, Dense, LayerNormalization, ReLU, Dropout, Input
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.backend import softmax 
import numpy as np
from numpy import random, array, savetxt, arange
from numpy.random import shuffle
import matplotlib.pyplot as plt
from pickle import load, dump, HIGHEST_PROTOCOL
from time import time

! wget https://github.com/Ploux/oe-nmt/raw/main/english-german-both.pkl


# Positional Embedding Layer

class PositionEmbeddingFixedWeights(Layer):
  def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
    super().__init__(**kwargs)
    word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
    pos_embedding_matrix = self.get_position_encoding(seq_length, output_dim)
    self.word_embedding_layer = Embedding(input_dim=vocab_size, output_dim=output_dim, weights=[word_embedding_matrix],trainable=False)
    self.position_embedding_layer = Embedding(input_dim=seq_length, output_dim=output_dim, weights=[pos_embedding_matrix],trainable=False)
  def get_position_encoding(self, seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
      for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[k, 2*i] = np.sin(k/denominator)
        P[k, 2*i+1] = np.cos(k/denominator)
    return P

  def call(self, inputs):
    position_indices = tf.range(tf.shape(inputs)[-1])
    embedded_words = self.word_embedding_layer(inputs)
    embedded_indices = self.position_embedding_layer(position_indices)
    return embedded_words + embedded_indices


# Dot Product Attention

class DotProductAttention(Layer):
  '''
  inherits from the Layer base class in Keras
  '''
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, queries, keys, values, d_k, mask=None):
    '''
    takes as input arguments the queries, keys, and values, as well as the dimensionality d_k , and a mask (that defaults to None )
    '''
  # Scoring the queries against the keys after transposing the latter, and scaling
    scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

    # Apply mask to the attention scores
    if mask is not None:
      scores += -1e9 * mask

    # Computing the weights by a softmax operation
    weights = softmax(scores)

    # Computing the attention by a weighted sum of the value vectors
    return matmul(weights, values)



