# download the corpus

#!wget https://github.com/ploux/oe-nmt/raw/main/corpus.tsv

# imports

import string
import re
import pandas as pd
from unicodedata import normalize
from pickle import load
from pickle import dump
from numpy import array
from numpy import argmax
from numpy.random import rand
from numpy.random import shuffle
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from keras.layers import Layer
import keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.pylab import plt

# clean data


class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


MAX_LENGTH = 20 # max num of words in eng and oe sentences

def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def filterPair(p):
    # print (p[0])
    # print (p[1])
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def to_pairs(doc):
  lines=doc.strip().split('\n')
  pairs=[line.split('\t') for line in lines]
  pairs = [pair for pair in pairs if filterPair(pair)]
  pairs = [list(reversed((p))) for p in pairs]
  return pairs

def clean_pairs(lines):
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

filename = 'corpus.tsv'
doc = load_doc(filename)
# print(doc)
pairs = to_pairs(doc)
clean_pairs2 = clean_pairs(pairs)
save_clean_data(clean_pairs2, 'oe-eng.pkl')

# print out 20 sentence pairs
for i in range(20):
    print('[%s] => [%s]' % (clean_pairs2[i,0], clean_pairs2[i,1]))
    
# train on 90%, test on 10%

dataset = load_clean_sentences('oe-eng.pkl')
shuffle(dataset)
pairs_length = len(pairs)
eighty_percent = int(len(pairs)*.8)
twenty_percent = pairs_length-eighty_percent

print(eighty_percent)
print(twenty_percent)

train, test = dataset[:eighty_percent], dataset[twenty_percent:]
save_clean_data(dataset, 'oe-eng-both.pkl')
save_clean_data(train, 'oe-eng-train.pkl')
save_clean_data(test, 'oe-eng-test.pkl')

# tokenize - break down into indivdual words

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

# encoder 
# Eng (source) integer-encoded
# OE (target) one-hot encoded

def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

# model

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(attention())
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

dataset = load_clean_sentences('oe-eng-both.pkl')
train = load_clean_sentences('oe-eng-train.pkl')
test = load_clean_sentences('oe-eng-test.pkl')

eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

oe_tokenizer = create_tokenizer(dataset[:, 1])
oe_vocab_size = len(oe_tokenizer.word_index) + 1
oe_length = max_length(dataset[:, 1])
print('Old English Vocabulary Size: %d' % oe_vocab_size)
print('Old English Max Length: %d' % (oe_length))

# encoding

trainX = encode_sequences(oe_tokenizer, oe_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

testX = encode_sequences(oe_tokenizer, oe_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# training

model = define_model(oe_vocab_size, eng_vocab_size, oe_length, eng_length, 256)
model.compile(optimizer='Adam', loss='categorical_crossentropy')
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

model_filename = 'model.h5'
checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
initial_learning_rate = 0.1
epochs = 200
decay = initial_learning_rate / epochs
def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

early_stopping_monitor = EarlyStopping(monitor='loss',min_delta = 1e-3,patience=1)

history_time_based_decay = model.fit(
    trainX, 
    trainY, 
    epochs=epochs, 
    validation_data=(testX, testY),
    batch_size=64,
    callbacks=[early_stopping_monitor, LearningRateScheduler(lr_time_based_decay, verbose=1), checkpoint],
)

plt.plot(list(range(1,len(history_time_based_decay.history['loss'])+1)), history_time_based_decay.history['loss'], label='Training Loss')
plt.savefig("Training Loss", dpi=300)
plt.clf()
plt.plot(list(range(1,len(history_time_based_decay.history['loss'])+1)), history_time_based_decay.history['val_loss'], label='Validation Loss')
plt.savefig("Validation Loss", dpi=300)

# evaluation

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 40:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append([raw_target.split()])
        predicted.append(translation.split())
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

filename = 'validate.tsv'
doc = load_doc(filename)
pairs = to_pairs(doc)
clean_pairs2 = clean_pairs(pairs)
save_clean_data(clean_pairs2, 'validate.pkl')

dataset = load_clean_sentences('oe-eng-both.pkl')
train = load_clean_sentences('oe-eng-train.pkl')
test = load_clean_sentences('oe-eng-test.pkl')
validate = load_clean_sentences('validate.pkl')

eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

oe_tokenizer = create_tokenizer(dataset[:, 1])
oe_vocab_size = len(oe_tokenizer.word_index) + 1
oe_length = max_length(dataset[:, 1])

trainX = encode_sequences(oe_tokenizer, oe_length, train[:, 1])
testX = encode_sequences(oe_tokenizer, oe_length, test[:, 1])
validateX = encode_sequences(oe_tokenizer, oe_length, validate[:, 1])

#model = load_model('model.h5')

print('Train:')
evaluate_model(model, eng_tokenizer, trainX, train)

print('Test:')
evaluate_model(model, eng_tokenizer, testX, test)

print("Validate:")
evaluate_model(model, eng_tokenizer, validateX, validate)