import numpy as np
import cv2
import os
import pickle

IMG_SIZE = (200,200)

DIR = r"C:\Users\chi_b\OneDrive\Desktop\Machine Learning\Flower Classifier Project\flowers_to_test_code"
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

#LOAD DATA

data = []

def import_data():
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        num_label = CATEGORIES.index(category)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img,IMG_SIZE)
            img = np.array(img, dtype=np.float32)
            
            data.append([img, num_label])
        
        #save data file:
        data_output= open('data_file.pickle', 'wb')
        pickle.dump(data,data_output)
        data_output.close()


import_data()

def load_data():
    data_input = open('data_file.pickle', 'rb')
    data = pickle.load(data_input)
    data_input.close()
    
    features = []
    labels = []
    
    for img, label in data:
        features.append(img)
        labels.append(label)
                              
                     
              
        
    features = np.array(features)
    labels = np.array(labels)
        
        #normalize data
    features = features/255
    
    return features, labels