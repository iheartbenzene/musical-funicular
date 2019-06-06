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
ap.add_argument('-e', '--encodings', required=True, help='path to serialized database of object encodings')
ap.add_argument('-d', '--detection-method', type=str, default='cnn', help='object detection method: hog or cnn')
args = vars(ap.parse_args())

try:
    data = pickle.loads(open(args['encodings'], 'rb').read())
    print('Image encodings loaded and ready to go!')
except:
    print('loading object analysis...')
    image_paths = list(paths.list_images(args['dataset']))
    encodings_accounted = []
    names_accounted = []

    for (i, image_path) in enumerate(image_paths):
        print('loading image analysis {}/{}...'.format(i+1, len(image_paths)))
        name = image_path.split(os.path.sep)[-2]

        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    binding_box = face_recognition.face_locations(rgb, model=args['detection_method'])
    encodings = face_recognition.face_encodings(rgb, binding_box)

    for encoding in encodings:
        encodings_accounted.append(encoding)
        names_accounted.append(name)

    print('...delicious serial...')
    data = {'encodings': encodings_accounted, 'names': names_accounted}

    with open(args['encodings'], 'wb') as f:
        f.write(pickle.dump(data))
    f.close()

image = cv2.imread(args['image'])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print('loading object analysis...')
# binding_box = face_recognition.face_locations(rgb, model=args['detection_method'])
# encodings = face_recognition.face_encodings(rgb, binding_box)

names = []

for encoding in encodings:
#    match_found = face_recognition.compare_faces(data['encodings'], encoding)
    name = 'As yet unknown'

    if True in match_found:
        match_index = [i for (i, b) in enumerate(match_found) if b]
        counts = {}

        for i in match_index:
            name = data['names'][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    names.append(name)
