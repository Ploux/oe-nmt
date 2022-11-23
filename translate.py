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

