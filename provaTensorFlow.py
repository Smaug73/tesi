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


#IMPORTANTE , CAMBIARE SUL PORTATILE LA VERSIONE DI PYTHON ALLA VERSINOE 3.6.1 , QUELLA UTILIZZATA SUL FISSO.

#Creaimo il primo layer del nostro modello di rete neurale
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),      #Questo primo layer serve solo a riformattare i pixel, disponendoli su di una unica dimensione
    keras.layers.Dense(128, activation='relu'),      #Questo secondo livello e collegato al primo ed e' formato da 128 neuroni
    keras.layers.Dense(10, activation='softmax')     #Questo terzo livello invece e' formato solo da 10 neuroni, cioe' il numero di classi che abbiamo
])                                                   #Tutti e 3 i livelli sono collegati in sequenza
                                                     #Ogni nodo dell'ultimo livello ritorna la probabilita' che l'immagine data in pasto alla rete neurale sia
                                                     #della classe rappresentata dal nodo.

#Prima di compilare il modello abbiamo bisogno di impostare:
#Loss function: serve a misurare l'accuratezza del modello ed a guidarlo al miglioramento
#Optimizer: indica come il modello viene aggiornato in base ai dati in ingresso e alla LossFunction
#Metrics: E' utilizzata per monitorare la fase di training e testing.In questo caso usiamo l'accuratezza, la frazione delle immagini correttamente classigicate.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Ora bisogna effettuare il  training della rete neurale.
#Per farlo abbiamo bisogno di passargli il training set e train label. Il modello imparera' ad associare il ogni immagina ad una label(etichetta)
model.fit(train_images, train_labels, epochs=10)


#Dopo aver effettuato il training del modello possiamo effettuare il testing di quest'ultimo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


#Dopo training e test possiamo provare a fare delle predizioni
predictions = model.predict(test_images)

#Effettuiamo una predizione
predictions[0]  #Una predizione e' un array di 10 numeri che rappresentano la confisenza che l'immagine corrispondente
                #possiede con ognuno delle 10 tipologie di articoli

np.argmax(predictions[0])#cosi' vediamo quale ha il valore di confidenza maggiore con l'immagine
                         #Cioe' vedremo a quale classe di piu' si avvicina l'immagine che gli passiamo

print('\nVALORE PREDETTO: ',np.argmax(predictions[0]))

#Per vedere se la predizione e' stata corretta basta vedere il valore di test_label[0]
print('\nVALORE VERO: ',test_labels[0])


#Costruiamo una funzione per graficare la predizione
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label: 
    color = 'blue'  #Se la predizione e' corretta coloriamo il nome di blue
  else:
    color = 'red'   #Altrimenti lo coloriamo di rosso

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#Effettuiamo la predizione del primo elemento di test
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)

plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


#Effettuiamo una predizione sulle prime 15 immagini del test set
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

