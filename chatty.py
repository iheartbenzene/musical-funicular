import numpy as np
import tensorflow as tf
import nltk
import json
import random
import pandas as pd
import pickle
import h5py

from pickle import dump, load
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import SGD

with open('json/intents1.json') as first_intent:
    data = json.load(first_intent)

try:
    with open('pickle/data.pkl', 'rb') as f:
        words, classes, train_x, train_y = load(f)
except:
    words = []
    classes = []
    docs = []
    ignore = ['?']

    for intent in data['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            docs.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [LancasterStemmer().stem(w.lower()) for w in words if w not in ignore]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(docs), "documents")
    print(len(classes), "classes")
    print(len(words), "stemmed words")

    training = []
    output_null = [0] * len(classes)

    for doc in docs:
        bag = []
        word_pattern = doc[0]
        word_pattern = [LancasterStemmer().stem(word.lower()) for word in word_pattern]
        for w in words:
            if w in word_pattern:
                bag.append(1)
            else:    
                bag.append(0)

        output_row = list(output_null)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    train_x, train_y = list(training[:,0]), list(training[:,1])

    print("\n Saving data... \n")

    with open('pickle/data.pkl', 'wb') as f:
        dump((words, classes, train_x, train_y), f)


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print("*np.array(train_x)[:1]", *np.array(train_x)[:1], "\n")
# print("np.array(train_x[0])", np.array(train_x[0]), "\n")
# print("np.array(train_x)[0]", np.array(train_x)[0], "\n")

# print(np.array(train_x).shape)
# print("np.array(train_x).dtype", np.array(train_x).dtype)
# print("np.array(train_x).shape", np.array(train_x).shape)

try:
    model = load_model('model/chatty.h5')
    print("\n Loaded Chatty... \n")
except:
    print("\n Fitting Model... \n")
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    print("\n Saving model to disk... \n")
    model.save('model/chatty.h5')

def bag_of_words(query, words):
    bag = [0 for _ in range(len(words))]

    query_words = nltk.word_tokenize(query)
    query_words = [LancasterStemmer.stem(word.lower()) for word in query_words]

    for r in query_words:
        for i, j in enumerate(words):
            if j == r:
                bag[i] = 1

    return np.array(bag)

def sentence_classification(sentence):
    THRESHOLD = 0.25

    input_data = pd.DataFrame([bag_of_words(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    results = [[i, r] for i, r in enumerate(results) if r > THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in results:
        result_list.append((classes[r[0]], str(r[1])))

    return result_list


def chat():
    print('Hello! What would you like to talk about?')
    while True:
        query = input(">>> ")
        if query.lower() == 'exit':
            break
        
        results = model.predict([bag_of_words(query, words)])
        results_index = np.argmax(results)
        tag = classes[results_index]

        for tags in data['intents']:
            if tags['tag'] == tag:
                responses = tags['responses']

        print(random.choice(responses))
