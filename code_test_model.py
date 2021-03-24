# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:44:39 2021

@author: chi_b
"""


from code_test import load_data

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,InputLayer,Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

VAL_SPLIT = 0.4
BATCH_SIZE = 10
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

(features, labels) = load_data()


#split data
(x_train, x_test, y_train, y_test) = train_test_split(features,labels, test_size =0.2)


#--------------------------------------Simple Model Building----------------------------------------------------

#callback
checkpoint = ModelCheckpoint('weights.h5', monitor = 'val_loss', save_best_only=True)
callbacks_list = [checkpoint]

input_layer = Input([200,200,3])

conv1 = Conv2D(10, kernel_size = (5,5), activation = 'relu', padding = 'same')(input_layer)

pool1 = MaxPooling2D(2)(conv1)

dropout1 = Dropout(0.25)(pool1)

conv2 = Conv2D(10, kernel_size = (5,5), activation = 'relu', padding = 'same')(pool1)

pool2 = MaxPooling2D(2)(conv2)

dropout2 = Dropout(0.25)(pool2)

flatten = Flatten()(dropout2)

output_layer = Dense(5, activation = 'softmax')(flatten)

model = Model(input_layer, output_layer)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, validation_split = VAL_SPLIT, epochs = 4, batch_size =BATCH_SIZE,callbacks=callbacks_list)

#evaluate the model
model.evaluate(x_test,y_test)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

#loading the stored parameters:
model.load_weights('weights.h5')
