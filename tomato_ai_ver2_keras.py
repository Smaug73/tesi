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

epochs=20


image_generator = ImageDataGenerator(rescale=1./255,validation_split=0.3)

data_gen = image_generator.flow_from_directory(directory=data_dir,
                                                           shuffle=True,
                                                           class_mode='binary',
                                                           subset='training')


test_gen= image_generator.flow_from_directory(directory=data_dir,
                                                           shuffle=True,
                                                           class_mode='binary',
                                                           subset='validation')

sample_training_images, _ = next(data_gen)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:10])

'''
model = Sequential([
    keras.layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers. MaxPooling2D(),
    keras.layers.Flatten( ),      #Questo primo layer serve solo a riformattare i pixel, disponendoli su di una unica dimensione
    keras.layers.Dense(64, activation='relu'),      #Questo secondo livello e collegato al primo ed e' formato da 128 neuroni
    keras.layers.Dense(10, activation='softmax')
])
'''
'''
model = Sequential([
    Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    keras.layers.Flatten( ),      #Questo primo layer serve solo a riformattare i pixel, disponendoli su di una unica dimensione
    keras.layers.Dense(510, activation='relu'),      #Questo secondo livello e collegato al primo ed e' formato da 128 neuroni
    keras.layers.Dense(10, activation='softmax')
])
'''

#Costruiamo una architettura che possieda i livelli di dropout, in modo da evitare l'overfitting
#Aumenteremo inoltre il numero di epoche.Questa tipologia sarà l'architettura definitiva della tesi(per ora).
#AGGIUNGERE batch normalization, riscala l'ordine di grandezza dell'ordine dei pesi interni, da inserire o come primo livello o prima dei dense
model = Sequential([
    keras.layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    keras.layers.MaxPooling2D(),   
    keras.layers.Dropout(0.2),                      #Un livello di dropout dopo il primo maxPooling 
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers. MaxPooling2D(),
    keras.layers.Dropout(0.2),                      #Un livello di dropout dopo l'ultimo livello di maxPooling 
    keras.layers.Flatten( ),                        #Questo primo layer serve solo a riformattare i pixel, disponendoli su di una unica dimensione
    keras.layers.BatchNormalization(),              #Aggiunto il batchNormalization
    keras.layers.Dense(64, activation='relu'),      #Questo secondo livello e collegato al primo ed e' formato da 128 neuroni
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',                         #provare ad utilizzare sgd con learning rate elevato e provarlo con diversi paramentri
              loss='sparse_categorical_crossentropy',   #usare la libreria hyperopt per modificare i parametri e fare più prove, iniziare con un solo parametro per evita di perdere il controllo
              metrics=['accuracy'])                     

model.summary()

history = model.fit_generator(
    data_gen,
    #steps_per_epoch=total_train #batch_size, numero totale di steps(lotti di campioni) da produrre dal generatore prima di dichiarare una ephoch terminata e iniziare la prossima epoch
    epochs=epochs,
    #verbose=1,
    validation_data=test_gen,
    #validation_steps=total_val  #batch_size
)

# Save the model
#history.save('C:\\Users\\stefr\\Desktop\\TESI DATASET\\Modelli\\model_v2.h5')
model.save('C:\\Users\\stefr\\Desktop\\TESI DATASET\\Modelli\\model_v_definitiva_3.h5')

#print(history.history[])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

