import cv2
import os
import numpy as np
import face_recognition
import dlib
import argparse
import pickle

from imutils import paths
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--dataset', required=True, help='path to the input directory of faces and images')
ap.add_argument('-e', '--encodings', required=True, help='path to serialized database of facial encodings')
ap.add_argument('-d', '--detection-method', type=str, default='cnn', help='facial detection method: hog or cnn')
args = vars(ap.parse_args())

print('analyzing faces...')
image_paths = list(paths.list_images(args['dataset']))
encodings_accounted = []
names_accounted = []

