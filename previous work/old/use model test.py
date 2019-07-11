'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
'''
from __future__ import print_function


import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model
import time
import pickle
import copy



BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'normal_abnormal')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')


embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'wiki.zh.vec')) as f:# , encoding="utf-8") as f:
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
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)  #MAX_NUM_WORDS = 20000
tokenizer.fit_on_texts(texts)  #--- need to consider Chinese tokenization
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

"""Pads each sequence to the same length => data 600 row, 1000 col
"""
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)# MAX_SEQUENCE_LENGTH = 1000
''' E.g. for use with categorical_crossentropy.'''
labels = to_categorical(np.asarray(labels))


# print('-------------------------------------------------------------------1')
# print(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# print('-------------------------------------------------------------------2')
# print(labels)

#VALIDATION_SPLIT = 0.2
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
# =======================================================================================


model = load_model('./model.h5')

print('Info: Successfully read model')

y_predicted = model.predict(x = x_val, batch_size=None, verbose=1, steps=None)

print('Info: Successfully get y_predicted')

# resultDict = {np.array([0, 1]).tostring():[], 
#               np.array([1, 0]).tostring():[]}

resultDict = {}

for i in range(0, len(x_val)):
    # print(y_predicted[i], '--', y_val[i])
    resultDict[tuple(y_val[i].tolist())] = resultDict.get(tuple(y_val[i].tolist()), [])
    resultDict[tuple(y_val[i].tolist())].append(y_predicted[i])
    # resultDict[y_val[i].tostring()].append(y_predicted[i])


for key in resultDict:
    base = copy.deepcopy(resultDict[key][0])
    for res in resultDict[key]:
        base += res
    base -= resultDict[key][0]
    base = base/len(resultDict[key])
    print(key, '--', base)

