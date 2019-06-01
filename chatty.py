import numpy as np
import tensorflow as tf
import nltk
import json
import random
import pandas as pd
import pickle

from pickle import dump, load
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import SGD

try:
    with open('data.pkl', 'rb') as f:
        words, classes, training, output = load(f)

except:
    with open('intents1.json') as first_intent:
        data = json.load(first_intent)

    words = []
    classes = []
    docs_x = []
    docs_y = []
    ignore = ['?']

    for intent in data['intents']:
        for pattern in intent['pattern']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent['tag'])

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [LancasterStemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    classes = sorted(classes)

    training = []
    output = []

    output_null = [0 for _ in range(len(classes))]

    for s, doc in enumerate(docs_x):
        bag = []
        word = [LancasterStemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in words:
                bag.append(1)
            bag.append(0)

        output_row = output_null[:]
        output_row[classes.index(docs_y[x])] = 1

        training.append([bag, output_row])

        # training.append(bag)
        # output.append(output_row)

    training, output = np.array(training), np.array(output)



    with open('data.pkl', 'wb') as f:
        dump((words, classes, training, output), f)

tf.reset_default_graph()

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

# try:
#     model.load('model.tflearn')
# except:
#     model.train(training, output, n_epoch=1e3, batch_size=8, show_metric=True)
#     model.save('model/model.tflearn')

def bag_of_words(query, words):
    bag = [0 for _ in range(len(words))]

    query_words = nltk.word_tokenize(query)
    query_words = [LancasterStemmer.stem(word.lower()) for word in query_words]

    for r in query_words:
        for i, j in enumerate(words):
            if j == r:
                bag[i] = 1

    return np.array(bag)

# def chat():
#     print('Hello! What would you like to talk about?')
#     while True:
#         query = input(">>> ")
#         if query.lower() == 'exit':
#             break
        
#         results = model.predict([bag_of_words(query, words)])
#         results_index = np.argmax(results)
#         tag = classes[results_index]

#         for tags in data['intents']:
#             if tags['tag'] == tag:
#                 responses = tags['responses']

#         print(random.choice(responses))

# chat()