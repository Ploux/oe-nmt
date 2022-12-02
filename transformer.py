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

## Encoder Layer
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

## Implementing the Encoder
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
    
# Decoder

## Decoder Layer
class DecoderLayer(Layer):
  def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
    super().__init__(**kwargs)
    self.build(input_shape=[None, sequence_length, d_model])
    self.d_model = d_model
    self.sequence_length = sequence_length
    self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
    self.dropout1 = Dropout(rate)
    self.add_norm1 = AddNormalization()
    self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
    self.dropout2 = Dropout(rate)
    self.add_norm2 = AddNormalization()
    self.feed_forward = FeedForward(d_ff, d_model)
    self.dropout3 = Dropout(rate)
    self.add_norm3 = AddNormalization()
  
  def build_graph(self):
    input_layer = Input(shape=(self.sequence_length, self.d_model))
    return Model(inputs=[input_layer], outputs=self.call(input_layer, input_layer, None, None, True))
    
  def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
    # Multi-head attention layer
    multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in a dropout layer
    multihead_output1 = self.dropout1(multihead_output1, training=training)

    # Followed by an Add & Norm layer
    addnorm_output1 = self.add_norm1(x, multihead_output1)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Followed by another multi-head attention layer
    multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)

    # Add in another dropout layer
    multihead_output2 = self.dropout2(multihead_output2, training=training)

    # Followed by another Add & Norm layer
    addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)
    
    # Followed by a fully connected layer
    feedforward_output = self.feed_forward(addnorm_output2)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in another dropout layer
    feedforward_output = self.dropout3(feedforward_output, training=training)

    # Followed by another Add & Norm layer
    return self.add_norm3(addnorm_output2, feedforward_output)

## Implementing the Decoder
class Decoder(Layer):
  def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
    super().__init__(**kwargs)
    self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
    self.dropout = Dropout(rate)
    self.decoder_layer = [DecoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
    
  
  def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
    # Generate the positional encoding
    pos_encoding_output = self.pos_encoding(output_target)
    # Expected output shape = (number of sentences, sequence_length, d_model)
    
    # Add in a dropout layer
    x = self.dropout(pos_encoding_output, training=training)
    
    # Pass on the positional encoded values to each encoder layer
    for i, layer in enumerate(self.decoder_layer):
      x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

    return x


# Transformer Model
class TransformerModel(Model):
  def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
    super().__init__(**kwargs)

    # Set up the encoder
    self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

    # Set up the decoder
    self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

    # Define the final dense layer
    self.model_last_layer = Dense(dec_vocab_size)

  def padding_mask(self, input):
    # Create mask which marks the zero padding values in the input by a 1.0
    mask = math.equal(input, 0)
    mask = cast(mask, float32)

    # The shape of the mask should be broadcastable to the shape of the attention weights that it will be masking later on
    return mask[:, newaxis, newaxis, :]

  def lookahead_mask(self, shape):
    # Mask out future entries by marking them with a 1.0
    mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

    return mask

  def call(self, encoder_input, decoder_input, training):
    # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
    enc_padding_mask = self.padding_mask(encoder_input)

    # Create and combine padding and look-ahead masks to be fed into the decoder
    dec_in_padding_mask = self.padding_mask(decoder_input)
    dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
    dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

    # Feed the input into the encoder
    encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

    # Feed the encoder output into the decoder
    decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

    # Pass the decoder output through a final dense layer
    model_output = self.model_last_layer(decoder_output)

    return model_output
    
    
# Prepare Dataset
class PrepareDataset:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = 10000  # Number of sentences to include in the dataset
        self.train_split = 0.9  # Ratio of the training data split

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)

        return len(tokenizer.word_index) + 1

    def __call__(self, filename, **kwargs):
        # Load a clean dataset
        clean_dataset = load(open(filename, 'rb'))

        # Reduce dataset size
        dataset = clean_dataset[:self.n_sentences, :]

        # Include start and end of string tokens
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

        # Random shuffle the dataset
        shuffle(dataset)

        # Split the dataset
        train = dataset[:int(self.n_sentences * self.train_split)]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(train[:, 0])
        enc_seq_length = self.find_seq_length(train[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        # Encode and pad the input sequences
        trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
        trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
        trainX = convert_to_tensor(trainX, dtype=int64)

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer(train[:, 1])
        dec_seq_length = self.find_seq_length(train[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        # Encode and pad the input sequences
        trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
        trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
        trainY = convert_to_tensor(trainY, dtype=int64)

        return (trainX, trainY, train, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)


# Train Model

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        # Linearly increasing the learning rate for the first warmup_steps, and
        # decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

# Prepare the training and test splits of the dataset
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, \
    enc_vocab_size, dec_vocab_size = dataset('english-german-both.pkl')

# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                  dec_seq_length, h, d_k, d_v, d_model, d_ff, n,
                                  dropout_rate)

# Defining the loss function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the
    # computation of loss
    mask = math.logical_not(equal(target, 0))
    mask = cast(mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(mask)

# Defining the accuracy function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the
    # computation of accuracy
    mask = math.logical_not(equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    mask = cast(mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(mask)

# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)
    
start_time = time()
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))

    

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f"Epoch {epoch+1} Step {step} Loss {train_loss.result():.4f} "
                  + f"Accuracy {train_accuracy.result():.4f}")

    # Print epoch number and loss value at the end of every epoch
    print(f"Epoch {epoch+1}: Training Loss {train_loss.result():.4f}, "
          + f"Training Accuracy {train_accuracy.result():.4f}")

    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print(f"Saved checkpoint at epoch {epoch+1}")

print("Total time taken: %.2fs" % (time() - start_time))
