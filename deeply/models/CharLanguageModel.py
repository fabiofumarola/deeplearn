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

shakespeare_base_directory = "shakespeare_lang_model"


def save_shakespeare_model(model,
                           text_window, slide, len_chars, char_indices, indices_char):
    model_path = os.path.join(shakespeare_base_directory, "shakespeare_model.json")

    model_obj = {
        "name": shakespeare_base_directory,
        "model": model.to_json(),
        "text_window": text_window,
        "slide": slide,
        "len_chars": len_chars,
        "char_indices": char_indices,
        "indices_char": indices_char
    }

    with open(model_path, 'w') as f:
        json.dump(model_obj, f)

    print("saved model at", model_path)

    return model_path


def load_model(model_path):

    with open(model_path, 'r') as f:
        model_obj = json.load(f)

    print("reloading", model_obj.name)

    model = model_from_json(model_obj.model)

    return (model, model_obj.text_window, model_obj.slide, model_obj.len_chars,
            model_obj.char_indices, model_obj.indices_char)

def train_shakespeare(epochs):

    if not os.path.exists(shakespeare_base_directory):
        os.makedirs(shakespeare_base_directory)

    path = get_file('shakespeare.txt', origin="http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt")
    text = open(path).read().lower()
    print('corpus length:', len(text))
    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = 100
    step = 3
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

    split_at = len(X) - len(X) / 20
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print(model.summary())

    save_shakespeare_model(model,
                           maxlen, step, len(chars), char_indices, indices_char)

    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

    tensorboard_dir = os.path.join(shakespeare_base_directory, "tensorboard_log")
    tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0)

    checkpoint_path = os.path.join(shakespeare_base_directory, "model_weights.{epoch:06d}-{val_loss:.4f}.hdf5")
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                   monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

    for i in range(epochs):
        print('-' * 50)
        print('Epoch ', i)
        model.fit(X_train, y_train, batch_size=128, nb_epoch=1, show_accuracy=True,
                  validation_data=(X_val, y_val),
                  callbacks=[checkpointer, early_stop, tensorboard])
        if i % 5 == 0:
            print()
            seed = "however, who have"
            for diversity in [0.2, 0.5, 0.8, 1.0]:
                generated_text = predict_shakespeare(model,
                                                     maxlen, step, chars, char_indices, indices_char,
                                                     seed, 1024, diversity)
                print(generated_text)


def sample(a, temperature=1.0):
    """helper function to sample an index from a probability array"""
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def predict_shakespeare(model,
                        text_window, slide, len_chars, char_indices, indices_char,
                        seed, length=1024, diversity=0.5):
    """
    :param seed: the base text used to start the generation
    :param length: the length of the generated text
    :param diversity: a value used to renormalize the inputs. A higher value selects
            less likely choices.
    :return:
        - info: the info for the generated text
        - generated: the generated text
    """
    sentence = seed

    info = '\n'
    info += '----- diversity: %s \n ' % diversity
    info += '----- Generating with seed: %s \n' % sentence

    generated = ''
    generated += sentence

    for i in range(length):
        x = np.zeros((1, text_window, len(len_chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        # sliding the sentence of a character character
        sentence = sentence[1:] + next_char

    generated += '\n'

    return info, generated


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("start the training for shakespeare passing the number of epochs \n"
              "python CharLanguageModel.py 200")
        exit(1)

    epochs = int(sys.argv[1])
    train_shakespeare(epochs)
