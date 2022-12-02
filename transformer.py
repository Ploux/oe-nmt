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
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, queries, keys, values, d_k, mask=None):
    # Scoring the queries against the keys after transposing the latter, and scaling
    scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

    # Apply mask to the attention scores
    if mask is not None:
      scores += -1e9 * mask

    # Computing the weights by a softmax operation
    weights = softmax(scores)

    # Computing the attention by a weighted sum of the value vectors
    return matmul(weights, values)


# Multi-Head Attention

class MultiHeadAttention(Layer):
  def __init__(self, h, d_k, d_v, d_model, **kwargs):
    super().__init__(**kwargs)
    self.attention = DotProductAttention() # Scaled dot product attention
    self.heads = h # Number of attention heads to use
    self.d_k = d_k # Dimensionality of the linearly projected queries and keys
    self.d_v = d_v # Dimensionality of the linearly projected values
    self.d_model = d_model # Dimensionality of the model
    self.W_q = Dense(d_k) # Learned projection matrix for the queries
    self.W_k = Dense(d_k) # Learned projection matrix for the keys
    self.W_v = Dense(d_v) # Learned projection matrix for the values
    self.W_o = Dense(d_model) # Learned projection matrix for the multi-head output

  def reshape_tensor(self, x, heads, flag):
    if flag:
      # Tensor shape after reshaping and transposing:
      # (batch_size, heads, seq_length, -1)
      x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
      x = transpose(x, perm=(0, 2, 1, 3))
    else:
      # Reverting the reshaping and transposing operations:
      # (batch_size, seq_length, d_k)
      x = transpose(x, perm=(0, 2, 1, 3))
      x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
    return x

  def call(self, queries, keys, values, mask=None):
    # Rearrange the queries to be able to compute all heads in parallel
    q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Rearrange the keys to be able to compute all heads in parallel
    k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
    
    # Rearrange the values to be able to compute all heads in parallel
    v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
    
    # Compute the multi-head attention output using the reshaped queries, keys, and values
    o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
    
    # Rearrange back the output into concatenated form
    output = self.reshape_tensor(o_reshaped, self.heads, False)
    # Resulting tensor shape: (batch_size, input_seq_length, d_v)
    
    # Apply one final linear projection to the output to generate the multi-head attention.
    # Resulting tensor shape: (batch_size, input_seq_length, d_model)
    return self.W_o(output)


# Encoder

## Add & Norm Layer
class AddNormalization(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.layer_norm = LayerNormalization() # Layer normalization layer
  
  def call(self, x, sublayer_x):
    # The sublayer input and output need to be of the same shape to be summed
    add = x + sublayer_x

    # Apply layer normalization to the sum
    return self.layer_norm(add)

## Feed-Forward Layer
class FeedForward(Layer):
  def __init__(self, d_ff, d_model, **kwargs):
    super().__init__(**kwargs)
    self.fully_connected1 = Dense(d_ff) # First fully connected layer
    self.fully_connected2 = Dense(d_model) # Second fully connected layer
    self.activation = ReLU() # ReLU activation layer
    
  def call(self, x):
    # The input is passed into the two fully-connected layers, with a ReLU in between
    x_fc1 = self.fully_connected1(x)

    return self.fully_connected2(self.activation(x_fc1))

# Encoder Layer
class EncoderLayer(Layer):
  def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
    super().__init__(**kwargs)
    self.build(input_shape=[None, sequence_length, d_model])
    self.d_model = d_model
    self.sequence_length = sequence_length
    self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
    self.dropout1 = Dropout(rate)
    self.add_norm1 = AddNormalization()
    self.feed_forward = FeedForward(d_ff, d_model)
    self.dropout2 = Dropout(rate)
    self.add_norm2 = AddNormalization()

  def build_graph(self):
    input_layer = Input(shape=(self.sequence_length, self.d_model))
    return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))
    
  def call(self, x, padding_mask, training):
    # Multi-head attention layer
    multihead_output = self.multihead_attention(x, x, x, padding_mask)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in a dropout layer
    multihead_output = self.dropout1(multihead_output, training=training)
    
    # Followed by an Add & Norm layer
    addnorm_output = self.add_norm1(x, multihead_output)
    # Expected output shape = (batch_size, sequence_length, d_model)
    
    # Followed by a fully connected layer
    feedforward_output = self.feed_forward(addnorm_output)
    # Expected output shape = (batch_size, sequence_length, d_model)
    
    # Add in another dropout layer
    feedforward_output = self.dropout2(feedforward_output, training=training)
    
    # Followed by another Add & Norm layer
    return self.add_norm2(addnorm_output, feedforward_output)

# Implementing the Encoder
class Encoder(Layer):
  def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
    super().__init__(**kwargs)
    self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
    self.dropout = Dropout(rate)
    self.encoder_layer = [EncoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

  def call(self, input_sentence, padding_mask, training):
    # Generate the positional encoding
    pos_encoding_output = self.pos_encoding(input_sentence)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in a dropout layer
    x = self.dropout(pos_encoding_output, training=training)

    # Pass on the positional encoded values to each encoder layer
    for i, layer in enumerate(self.encoder_layer):
      x = layer(x, padding_mask, training)

    return x