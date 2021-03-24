# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:44:39 2021

@author: chi_b
"""


from import_data import load_data

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,InputLayer,Flatten,Dropout,Conv2D,MaxPooling2D, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

DIR = r"C:\Users\chi_b\OneDrive\Desktop\Machine Learning\Flower Classifier Project"

VAL_SPLIT = 0.3
BATCH_SIZE = 15
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

IMG_ROWS = 150
IMG_COLS = 150

features, labels = load_data()

#split data
(x_train, x_test, y_train, y_test) = train_test_split(features,labels, test_size =0.2)



#--------------------------------------Simple Model Building (no augmentations)----------------------------------------------------

#callback

def fit_model(lr):
    checkpoint = ModelCheckpoint('weights.h5', monitor = 'val_loss', save_best_only=True)
    early_stopping_monitor = EarlyStopping(patience=2)
    callbacks_list = [checkpoint, early_stopping_monitor]
    
    opt = Adam(learning_rate=lr)
    
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (150,150,3)))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(20, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(20, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(20, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(20, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(5, activation = 'softmax'))
    
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    
    #-----------------------------------FIT AND EVALUATE THE MODEL--------------------------------------------------------------------
    history = model.fit(x_train, y_train, validation_split = VAL_SPLIT, epochs = 5, batch_size =BATCH_SIZE,callbacks=callbacks_list)

    #graph the Loss by epochs
    
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title('lrate='+str(lr), pad=-50)


learning_rates = [0.0001,0.001,0.1,1]

for i in range(len(learning_rates)):
	# determine the plot number
	plot_no = 420 + (i+1)
	plt.subplot(plot_no)
	# fit model and plot learning curves for a learning rate
	fit_model(learning_rates[i])
# show learning curves
plt.show()    









#loading the stored parameters:
#model.load_weights('weights.h5')


#Save the Model
#model.save(DIR + 'model.h5')