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

#Dettagli immagini utilizzate
IMG_HEIGHT=256
IMG_WIDTH=256


image_generator = ImageDataGenerator(rescale=1./255,validation_split=0.3)

data_gen = image_generator.flow_from_directory(directory=data_dir,
                                                           shuffle=True,
                                                           class_mode='binary')


sample_training_images, _ = next(data_gen)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    data_gen,
    #steps_per_epoch=total_train #batch_size, numero totale di steps(lotti di campioni) da produrre dal generatore prima di dichiarare una ephoch terminata e iniziare la prossima epoch
    epochs=15,
    #validation_data=val_data_gen,
    #validation_steps=total_val  #batch_size
)