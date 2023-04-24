# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:37:57 2023

@author: Suraj Raj
"""

import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

loaded_model=pickle.load(open('C:/Users/Suraj Raj/Desktop/Brain Tumor/brain_tumor.sav','rb'))


my_image = load_img('C:/Users/Suraj Raj/Desktop/Brain Tumor/Dataset/tumor.jpg', target_size=(128, 128))
my_image = img_to_array(my_image)
my_image = my_image/255
my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
prediction = loaded_model.predict(my_image)
print(prediction)
predict_index = np.argmax(prediction)
if(predict_index==0):
    print("Glioma")
elif(predict_index==1):
    print("meningioma")
elif(predict_index==2):
    print("notumer")
elif(predict_index==3):
    print("pituitary")    
else:
    print("not know")