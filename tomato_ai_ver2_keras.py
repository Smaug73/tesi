from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tomato_dataset import Dataset


#Librerie per keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir='C:\\Users\\stefr\\Desktop\\TESI DATASET\\PlantVillage'

image_generator = ImageDataGenerator(rescale=1./255,validation_split=0.3)

data_gen = image_generator.flow_from_directory(directory=data_dir,
                                                           shuffle=True,
                                                           
                                                           class_mode='binary')