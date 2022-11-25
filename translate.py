# download the corpus

!wget https://github.com/ploux/oe-nmt/raw/main/corpus.tsv

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
from keras.utils import to_categorical
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
from keras.utils import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

# clean data

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
clean_pairs = clean_pairs(pairs)
save_clean_data(clean_pairs, 'me-oe.pkl')

# print out 20 sentence pairs
for i in range(20):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
    
# train on 90%, test on 10%

dataset = load_clean_sentences('me-oe.pkl')
shuffle(dataset)
pairs_length = len(pairs)
ninety_percent = int(len(pairs)*.9)
ten_percent = pairs_length-ninety_percent

print(ninety_percent)
print(ten_percent)

train, test = dataset[:336], dataset[336:]
save_clean_data(dataset, 'me-oe-both.pkl')
save_clean_data(train, 'me-oe-train.pkl')
save_clean_data(test, 'me-oe-test.pkl')    