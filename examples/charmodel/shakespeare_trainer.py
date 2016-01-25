
# coding: utf-8

# In[ ]:

from __future__ import print_function
from keras.models import Sequential, slice_X
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
from keras.callbacks import *
import os.path
import json
from keras.models import model_from_json


# In[ ]:

def sample(a, temperature=1.0):
    """helper function to sample an index from a probability array"""
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# ### Parameters

# In[ ]:

maxlen = 50
step = 1
testsize = 0.05
layers = [256, 256, 256]
dropout = 0.2
iterations = 40
batch_size = 1000
save = False
checkpoint_dir = "shakespeare"
early_stop = 20
tensorboard = False


# In[ ]:

path = get_file('shakespeare.txt', origin="http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt")
text = open(path).read().lower()
print('corpus length:', len(text))


# In[ ]:

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[ ]:

# cut the text in semi-redundant sequences of maxlen characters
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[ ]:

split_at = int(len(X) - len(X) * testsize)
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])


# In[ ]:

model = Sequential()

for i, size in enumerate(layers):
    if i == 0:
        model.add(LSTM(size, return_sequences=True, input_shape=(maxlen, len(chars))))
    elif i < len(layers) -1:
        model.add(LSTM(size, return_sequences=True))
    else:
        model.add(LSTM(size, return_sequences=False))

    if dropout > 0:
        model.add(Dropout(dropout))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))


# In[ ]:

print('Build model...')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# In[ ]:

print(model.summary())


# In[ ]:

if save:
    print("Save the model...")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, "shakespeare_model.json")
    model_obj = {
        "name": "shakespeare",
        "model": model.to_json(),
        "maxlen": maxlen,
        "step": step,
        "char_indices": char_indices,
        "indices_char": indices_char
    }
    with open(model_path, 'w') as f:
        json.dump(model_obj, f)


# In[ ]:

callbacks = []

if save:
    checkpointfile = os.path.join(checkpoint_dir, "model_weights.{epoch:03d}-{val_loss:.4f}.hdf5")
    checkpointer = ModelCheckpoint(filepath=checkpointfile,
                              monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
    callbacks.append(checkpointer)

if early_stop > 0:
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=1, mode='auto')
    callbacks.append(early_stop)

if tensorboard:
    print("Tensorboard log ...")
    tensorboard_dir = os.path.join(checkpoint_dir, "tensorboard")
    callbacks.append(callbacks)


# In[ ]:

for i in range(0, iterations):
    print('-' * 50)
    print('Epoch ', i)
    model.fit(X, y, batch_size=batch_size, nb_epoch=1,
              show_accuracy=True, validation_data=(X_val,y_val),
             callbacks=callbacks)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(200):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()



# In[ ]:
