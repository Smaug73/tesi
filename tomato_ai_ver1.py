from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tomato_dataset import Dataset




tomato_dataset = Dataset('C:\\Users\\stefr\\Desktop\\TESI DATASET\\PlantVillage')

train_images, test_images = tomato_dataset.splitTrainingTest(0.7)

class_names = tomato_dataset.getClasses()


#esploriamo i dati
#tomato_dataset.showImage(class_names[1],1)
#train_images.showImage(class_names[1],1)
#test_images.showImage(class_names[1],1)

#Testiamo metodo nuovo
'''
img_list, label_list = train_images.createArrayImgs()
img_list_t, label_list_t = test_images.createArrayImgs()

print(img_list[0])
print(label_list[0])

print(len(img_list))
print(len(label_list))

print(len(img_list_t))
print(len(label_list_t))
'''

#Creazione dataset per tensorflow
'''
#creiamo il dataset partendo da dateset contente tutte le immagini, faremo train e test set successivamente
img_list,label_list= tomato_dataset.createArrayImgs()

#creiamo le liste contente i nomi(in questo caso i path) delle immagini e una lista per le label
img_name= tf.constant(img_list)
labels = tf.constant(label_list)

#creaimo il dataset
dataset = tf.data.Dataset.from_tensor_slices((img_name,labels))
#Il dataset può essere creato utilizzando anche il metodo list_files
#Con il metodo list_files è possibile passare la directory che continene gli elementi del dataset
#Nel mio caso ho più cartelle quindi bisognerebbe creare un dataset per ogni cartella(corrispondente alla classe) e poi 
#unire i vari dataset


'''
#effettuiamo il parsing di tutte le immagini nel dataset attraverso un mapping
#per questa parte ho visto meglio su https://www.tensorflow.org/guide/data#decoding_image_data_and_resizing_it
def _parse_function(img_name,label):
    img_string = tf.io.read_file(img_name)
    image= tf.io.decode_jpeg(img_string)
    
    #img_decoded = tf.image.decode_jpeg(img_string)
    #image = tf.image.convert_image_dtype(img_decoded, tf.float32)
    #convertiamo la scala nella scala dei grigi 
    #image = tf.image.rgb_to_grayscale(image)
    #image = image/255
    return image, label
'''
dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

'''
#test _parse_function
def show(image, label):
  plt.figure()
  plt.imshow(image)
  #plt.imshow(image, cmap=plt.cm.binary)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()

#test _parse_function
#i, l = _parse_function(img_name[0],labels[0])
#show(i,l)



#Una volta costruito il dataset proviamo ad utilizzarlo per la creazione di un 
# modello .
#Per prima cosa abbiamo bisogno di creare train e test set e di modificare le immagini per renderle utilizzabili dalla rete neurale.

train_path,label_tr =train_images.createArrayImgs()
test_path, label_te= test_images.createArrayImgs()


train= tf.constant(train_path)
test= tf.constant(test_path)
labels_tr = tf.constant(label_tr)
labels_te = tf.constant(label_te)

train_dataset= tf.data.Dataset.from_tensor_slices((train,labels_tr))
test_dataset= tf.data.Dataset.from_tensor_slices((test,labels_te))

#Aggiungiamo il mapping ai dataset
train_imgs = train_dataset.map(_parse_function)
test_imgs = test_dataset.map(_parse_function)

#testing dataset
for i,l in train_imgs.take(1):
    show(i,l)

'''



#Una volta ottenuti i due dataset creiamo il modello e passiamo al suo training. 

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
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#Ora bisogna effettuare il  training della rete neurale.
#Per farlo abbiamo bisogno di passargli il training set e train label. Il modello imparera' ad associare il ogni immagina ad una label(etichetta)
model.fit(train_imgs, epochs=10)

test_loss, test_acc = model.evaluate(test_imgs, verbose=2)

print('\nTest accuracy:', test_acc)

'''