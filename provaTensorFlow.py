#ESEMPIO FORONITO DA TENSOR FLOW PER LA CREAZIONE DI UN SEMPLICE CLASSIFICATORE DI IMMAGINI
#IN QUESTO CASO SI ANDRANNO A CLASSIFICARE I VESTITI PRESENTI IN UN DATASET SCARICABILE ONLINE 


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



fashion_mnist = keras.datasets.fashion_mnist
#CARICHIAMO IL DATASET CHE CI RESTITUIRA' 4 ARRAY DI NUMPY
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#train_images, train_labels sono gli array che serviranno per il training del nostro modello
#test_images, test_labels sono gli array che serviranno per il test del modello

#Array contenenti le varie classi di riferimento
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Esploriamo i dati contenuti nel dataset:
train_images.shape
len(train_labels)
train_labels

test_images.shape
len(test_labels)

#Per utilizzare un modello abbiamo bisogno di modificare le immagini effettuando un preprocess di esse
#Dobbiamo modificare il  valore dei pixel affinche' siano compresi tra 0 e 1
#Per fare cio' modifichiamo i pixel divisendoli per 255 dato che il suo valore varia da 0 a 255

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

#Verifichiamo le modifiche effettuate mostrando le prime 25 immagini del training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()