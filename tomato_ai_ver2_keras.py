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


from PIL import Image

data_dir='C:\\Users\\stefr\\Desktop\\TESI DATASET\\PlantVillage'

#Dettagli immagini utilizzate
IMG_HEIGHT=256
IMG_WIDTH=256

epochs=16  # per evitare overfitting

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

'''    
#Per risolvere il problema dello sbilanciamento delle classi, creare nuove immagini a partire da quelle della
# classe sbilanciata e salarle nella stessa directory
fix_class= 'C:\\Users\\stefr\\Desktop\\TESI DATASET\\classe-da-fixare'
class_gen = ImageDataGenerator(rescale=1./255,horizontal_flip=True, rotation_range=45, zoom_range=0.5)  #Creiamo un generatore di immagini che applica delle 
                                                                                                        #delle trasformazioni alle immagini
classe_sbilanciata_gen= class_gen.flow_from_directory(directory=fix_class,
                                                           shuffle=True,
                                                           class_mode='binary',
                                                           )   

images,_=next(classe_sbilanciata_gen)                                                                                                                                                               
#Verifichiamo che le immagini siano modificate correttamente
print(images.size)
saved_image= images[:2]
c=0
for i in saved_image:
    fix_i=tf.keras.preprocessing.image.array_to_img(i)
    nome= str(c)+'.png'
    fix_i.save(nome)
    c=c+1
#plotImages(saved_image[:20])

'''

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


plotImages(sample_training_images[:10])

'''
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
'''
#Costruiamo una architettura che possieda i livelli di dropout, in modo da evitare l'overfitting
#Aumenteremo inoltre il numero di epoche.Questa tipologia sarà l'architettura definitiva della tesi(per ora).
#AGGIUNGERE batch normalization, riscala l'ordine di grandezza dell'ordine dei pesi interni, da inserire o come primo livello o prima dei dense
model = Sequential([
    keras.layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    keras.layers.BatchNormalization(),              #Aggiunto il batchNormalization PROVA
    keras.layers.MaxPooling2D(),   
    keras.layers.Dropout(0.1),                      #Un livello di dropout dopo il primo maxPooling 

    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),              #Aggiunto il batchNormalization PROVA
    keras.layers. MaxPooling2D(),
    keras.layers.Dropout(0.1),                      #Un livello di dropout dopo l'ultimo livello di maxPooling 

    keras.layers.Flatten( ),                        #Questo primo layer serve solo a riformattare i pixel, disponendoli su di una unica dimensione
    keras.layers.BatchNormalization(),              #Aggiunto il batchNormalization
    keras.layers.Dense(64, activation='relu'),      #Questo secondo livello e collegato al primo ed e' formato da 128 neuroni
    keras.layers.Dense(10, activation='softmax')
])
'''
'''
model.compile(optimizer='adam',                         #provare ad utilizzare sgd con learning rate elevato e provarlo con diversi paramentri
              loss='sparse_categorical_crossentropy',   #usare la libreria hyperopt per modificare i parametri e fare più prove, iniziare con un solo parametro per evita di perdere il controllo
              metrics=['accuracy']) 
'''
'''
#Tipo di compilazione con SGD e learning rate elevato
model.compile(optimizer=keras.optimizers.SGD(),                         #provare ad utilizzare sgd con learning rate elevato e provarlo con diversi paramentri
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
model.save('C:\\Users\\stefr\\Desktop\\TESI DATASET\\Modelli\\model_v_definitiva_9.h5')

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

