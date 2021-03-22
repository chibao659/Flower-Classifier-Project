# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:48:31 2021

@author: chi_b
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


PATH = r'C:\Users\chi_b\OneDrive\Desktop\Machine Learning\Flower Classifier Project\flowers'

example_image = plt.imread(r"%s\daisy\134409839_71069a95d1_m.jpg" % PATH)

plt.imshow(example_image)
plt.show()

print(example_image)
print(example_image.shape)


plt.imshow(example_image[:,:,0])
plt.imshow(example_image[:,:,1])
plt.imshow(example_image[:,:,2])

plt.imshow(example_image[50:150,50:150,:])

example_image[:,:,0]=0
plt.imshow(example_image)
