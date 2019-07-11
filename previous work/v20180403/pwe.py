
from __future__ import print_function
import sys


F_NUMBER = int(sys.argv[1]) #1000
SEARCH_OR_ACCURATE = sys.argv[2] #'acc'
PUNC_OR_NOPUNC = sys.argv[3] #'noPunc'
EPOCHS = 20

'''
=============================================================================
ATTENTION!
dataset folder name = './500_search_punc/'
=============================================================================
'''



FILTERS = 64
OPTIMIZER = 'Adamax'
BATCH_SIZE = 40
ACTIVATION_CONV = 'relu'
INIT_MODE = 'uniform'


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
start = time.time()

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, str(F_NUMBER) + '_' + SEARCH_OR_ACCURATE + '_' + PUNC_OR_NOPUNC)
MAX_SEQUENCE_LENGTH = F_NUMBER
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
SAVE_DIR = str(F_NUMBER) + '_' + SEARCH_OR_ACCURATE + '_' + PUNC_OR_NOPUNC + '.model'

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
# print('sequences-------------------------------------------------------------------2')
# print(sequences)
# mipa
# [[199, 64, 2, 309, 364, 3860, ... 108, 417, 169, 893]
#  ...
#  [ 2, 361, 38, 209, 6, 4415, ... 1, 559, 24, 312, 4]]

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# print('word_index-------------------------------------------------------------------2')
# print(word_index)
# mipa
# { ... '\xe7\x89\xb5\xe6\x8c\x82': 5888, '\xe7\x85\xa7': 1262, '\xe7\x85\xae': 1532}

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
# print(indices)
# [0 1 2 3 ... 824 825]
# mipa
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


# print('data-------------------------------------------------------------------2')
# print(data)
print(len(data))
print(len(data[0]))
# mipa
# [[0 0 0 ... 18 25 174]
#  [0 0 7618 ... 3 2645 4]
#  ...
#  [0 0 0 ... 9 178 7]
#  [0 6 1 ... 3 3 384]]
# print('labels-------------------------------------------------------------------2')
# print(labels)
# mipa
# [[1. 0.]
#  [0. 1.]
#  [1. 0.]
#  ...
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]

#VALIDATION_SPLIT = 0.2
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# print('-------------------------------------------------------------------3')
# print(y_val)

print('Preparing embedding matrix.')
print('len(word_index): ', len(word_index))

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1) #len(word_index):  31702
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))# 20000 row, EMBEDDING_DIM = 300
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:    #skip low freqency words
        continue
    embedding_vector = embeddings_index.get(word)  #embeddings_index=>pretrained word vector
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,   #ouput dimention EMBEDDING_DIM = 300
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#print("embedding_layer.shape ",embedding_layer.shape())


print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') #MAX_SEQUENCE_LENGTH = 1000
embedded_sequences = embedding_layer(sequence_input)

print('embedded_sequences-------------------------------------------------------------------2')
print(embedded_sequences)
# mipa
# Tensor("embedding_1/Gather:0", shape=(?, 1000, 300), dtype=float32)

x = Conv1D(FILTERS, 5, kernel_initializer = INIT_MODE, activation = ACTIVATION_CONV)(embedded_sequences)

print('x-------------------------------------------------------------------2')
print(x)
# mipa FILTERS = 92
# Tensor("conv1d_1/Softplus:0", shape=(?, 996, 92), dtype=float32)

x = MaxPooling1D(5)(x)

print('x-------------------------------------------------------------------3')
print(x)
# mipa
# Tensor("max_pooling1d_1/Squeeze:0", shape=(?, 199, 92), dtype=float32)

x = Conv1D(FILTERS, 5, kernel_initializer = INIT_MODE, activation = ACTIVATION_CONV)(x)

print('x-------------------------------------------------------------------4')
print(x)
# mipa
# Tensor("conv1d_2/Softplus:0", shape=(?, 195, 92), dtype=float32)

x = MaxPooling1D(5)(x)

print('x-------------------------------------------------------------------5')
print(x)
# mipa
# Tensor("max_pooling1d_2/Squeeze:0", shape=(?, 39, 92), dtype=float32)

x = Conv1D(FILTERS, 5, kernel_initializer = INIT_MODE, activation = ACTIVATION_CONV)(x)

print('x-------------------------------------------------------------------6')
print(x)
# mipa
# Tensor("conv1d_3/Softplus:0", shape=(?, 35, 92), dtype=float32)

x = GlobalMaxPooling1D()(x)

print('x-------------------------------------------------------------------7')
print(x)
# mipa
# Tensor("global_max_pooling1d_1/Max:0", shape=(?, 92), dtype=float32)

x = Dense(FILTERS, kernel_initializer = INIT_MODE, activation='relu')(x)

print('x-------------------------------------------------------------------8')
print(x)
# mipa
# Tensor("dense_1/Relu:0", shape=(?, 92), dtype=float32)

preds = Dense(len(labels_index), kernel_initializer = INIT_MODE, activation='softmax')(x)

print('preds-------------------------------------------------------------------9')
print(preds)
# mipa
# Tensor("dense_2/Softmax:0", shape=(?, 2), dtype=float32)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer = OPTIMIZER,
              metrics=['acc'])




model.fit(x_train, y_train,
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          validation_data = None)#=(x_val, y_val))


print("****time elapsed: ", (time.time() - start))

model.save(SAVE_DIR)

print('Info: Successfully saved model')

# =======================================================================================


model = load_model(SAVE_DIR)

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

print('===================================================')
print('class', '--', 'predicted average')

accNum = 0
totalNum = 0

for key in resultDict:
    base = copy.deepcopy(resultDict[key][0])
    for res in resultDict[key]:
        base += res
        
        if key.index(max(key)) == res.tolist().index(max(res.tolist())):
            accNum += 1
        totalNum += 1

    base -= resultDict[key][0]
    base = base/len(resultDict[key])
    print(key, '--', base)

print('Accuracy =', float(accNum)/float(totalNum))


