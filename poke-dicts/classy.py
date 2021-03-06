import numpy as np
import matplotlib.pyplot as plt
import scipy
import datetime
import os
import h5py

from keras.datasets import cifar10
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
# from scipy.misc import toimage

# K.set_image_dim_ordering('th')

def initial_model(seed):
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(scipy.misc.toimage(train_x[i]))
    # plt.show()

    train_x = train_x.astype('float32')
    train_x = train_x / 255.0
    test_x = test_x.astype('float32')
    test_x = test_x / 255.0

    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)
    number_of_classes = test_y.shape[1]

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), padding = 'same', activation='relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding = 'same', activation='relu', kernel_constraint = maxnorm(3)))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Conv2D(64, (3, 3), padding = 'same', activation='relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding = 'same', activation='relu', kernel_constraint = maxnorm(3)))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Conv2D(128, (3, 3), padding = 'same', activation='relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding = 'same', activation='relu', kernel_constraint = maxnorm(3)))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))

    epochs = 50
    # epochs = 1
    learing_rate = 0.01
    decay = learing_rate/epochs
    sgd = SGD(lr=learing_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    np.random.seed(seed)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=64)
    scores = model.evaluate(test_x, test_y, verbose=0)
    print('Accuracy: %0.3f%%' % (scores[1]*100))
    print(datetime.datetime.now().strftime('%Y%m%d-%H%m'))
    model.save(filepath=os.path.abspath('model/classy.h5'), overwrite=True)

try:
    classification_model = None
    model = Sequential()
    classification_module = load_model(os.path.abspath('model/classy.h5'))
    print("\n Loaded Classification Module... \n")
except:
    print("\n Fitting Classification Model... \n")
    initial_model(7)
    print("\n Saving classification model to disk... \n")