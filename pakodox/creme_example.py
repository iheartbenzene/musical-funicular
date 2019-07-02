import cv2
import imutils
import sklearn
import keras
import creme
import numpy as np
import argparse
import pickle
import random
import os

from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from imutils import path

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='input dataset')
ap.add_argument('-c', '--csv', required=True, help='output csv')
ap.add_argument('-b', '--batch-size', type=int, default=32, help='batch size')
arguments = vars(ap.parse_args())

model = ResNet50(weights='imagenet', include_top=False)
size_of_batch = arguments['batch_size']

image_paths = list(path.list_images(arguments['dataset']))
random.seed(42)
random.shuffle(image_paths)

labels = [point.split(os.path.sep)[-1].split('.') for point in image_paths]
label_encode = LabelEncoder()
labels = label_encode.fit_transform(labels)

columns = ['feature_{}'.format(i) for i in range (0, 7 * 7 * 2048)]
columns = ['class'] + columns