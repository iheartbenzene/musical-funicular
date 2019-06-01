import numpy as np
import tensorflow as tf
import tflearn as tfl
import nltk
import json

from pickle import dump, load
from nltk.stem.lancaster import LancasterStemmer

import tensorflow.contrib 

with open('intents1.json') as first_intent:
    data = json.load(first_intent)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['pattern']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs_x.append(word)
        docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [LancasterStemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

output_null = [0 for _ in range(len(labels))]

for s, doc in enumerate(docs_x):
    bag = []
    word = [LancasterStemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in words:
            bag.append(1)
        bag.append(0)

    output_row = output_null[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training, output = np.array(training), np.array(output)

tf.reset_default_graph()

net = tfl.input_data(shape=[None, len(training(0))])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(output[0]), activation='softmax')
net = tfl.regression(net)

model = tfl.DNN(net)

model.fit(training, output, n_epoch=1e3, batch_size=8, show_metric=True)
model.save('model/model.tflearn')