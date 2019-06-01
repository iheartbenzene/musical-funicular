import numpy as np
import tensorflow as tf
import tflearn as tfl
import nltk
import json

from pickle import dump, load
from nltk.stem.lancaster import LancasterStemmer

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