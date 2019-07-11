# Date: 2018/02/27

'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
'''
from __future__ import print_function

import time

start = time.time()


import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'normal_abnormal')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

# help(Conv1D)

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'wiki.zh.vec'), encoding="utf-8") as f:
    header = f.readline()
    # vocab_size == vocabulary size, dimension == word2vec dimension
    vocab_size, dimension = map(int, header.split())
    print(vocab_size)
    print(dimension)
    counter = 0

    for i in range(vocab_size):
        counter += 1
        line = f.readline()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    print(counter)

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for folderName in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, folderName)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[folderName] = label_id
        for fname in sorted(os.listdir(path)):
            # if fname.isdigit():
            fpath = os.path.join(path, fname)
            args = {} if sys.version_info < (3,) else {'encoding': 'utf-8'}
            with open(fpath, **args) as f:
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
            labels.append(label_id)
'''labels = [0 0 0 0... 1 1 1 1... ]'''

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)  # MAX_NUM_WORDS = 20000
tokenizer.fit_on_texts(texts)  # --- need to consider Chinese tokenization
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

"""Pads each sequence to the same length => data 600 row, 1000 col
"""
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # MAX_SEQUENCE_LENGTH = 1000
''' E.g. for use with categorical_crossentropy.'''
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# VALIDATION_SPLIT = 0.2
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')
print('len(word_index): ', len(word_index))

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)  # len(word_index):  31702
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  # 20000 row, EMBEDDING_DIM = 300
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:  # skip low freqency words
        continue
    embedding_vector = embeddings_index.get(word)  # embeddings_index=>pretrained word vector
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,  # ouput dimention EMBEDDING_DIM = 300
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# print("embedding_layer.shape ",embedding_layer.shape())


print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # MAX_SEQUENCE_LENGTH = 1000
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

# preds = Dense()(x)  # this might be the document vector


model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=30,
          validation_data=(x_val, y_val))

print("****time elapsed: ", (time.time() - start))