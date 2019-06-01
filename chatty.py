import numpy as np
import tensorflow as tf
import tflearn as tfl
import nltk
import json

'''
Known error:

<module>
    from tensorflow.contrib.framework.python.ops import add_arg_scope as contrib_add_arg_scope
ModuleNotFoundError: No module named 'tensorflow.contrib'

'''

from pickle import dump, load
from nltk.stem.lancaster import LancasterStemmer

try:
    with open('data.pkl', 'rb') as f:
        words, labels, training, output = load(f)

except:
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

    with open('data.pkl', 'wb') as f:
        dump((words, labels, training, output), f)

tf.reset_default_graph()

net = tfl.input_data(shape=[None, len(training(0))])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(output[0]), activation='softmax')
net = tfl.regression(net)

model = tfl.DNN(net)

try:
    model.load('model.tflearn')
except:
    model.fit(training, output, n_epoch=1e3, batch_size=8, show_metric=True)
    model.save('model/model.tflearn')

