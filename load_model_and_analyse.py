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

#class_names = ['Tomato_', 'Tomato_', 'Tomato_', 'Tomato_', 'Tomato_',
#               'Tomato_', 'Tomato_', 'Tomato_', 'Tomato_', 'Tomato_']




#Carichiamo il modello dalla directory nella quale e' stato salvato
model= tf.keras.models.load_model('C:\\Users\\stefr\\Desktop\\TESI DATASET\\Modelli\\model_v_definitiva_5.h5')


#Proviamo a effettuare delle predizioni utilizzando il modello caricato
data_dir='C:\\Users\\stefr\\Desktop\\TESI DATASET\\PlantVillage'

image_generator = ImageDataGenerator(rescale=1./255,validation_split=0.1)   #Effettuiamo delle predizioni utilizzando alcune foto prese casualmente da una cartella presa a parte dal dataset
                                                                            #Ho scelto un validation split piccolo in modo da prendere solo poche foto del dataset
pred_gen= image_generator.flow_from_directory(directory=data_dir,
                                                           shuffle=True,
                                                           class_mode='categorical',
                                                           subset='validation')
#IMPORTANTE PER LE LABELS: nel momento in cui carichiamo dalla directory ogni sua subdirectory come classe, le rispettive sottodirectory sono mappate
#                           in ordine alfanumerico, questo significa che alla classe 0 corrisponderà il nome della classe(che sarà il nome della sottodirectory)
#                           scelto in modo alfanumerico.Per ottenere il dizionario contenente il mapping tra nome delle classi(nome directory) e label(numeri) può
#                           può essere ottenuto prendendo l'attributo class_indices .

class_dict= pred_gen.class_indices
print(class_dict)
#Prendiamoci la lista dei nomi delle classi
class_names= np.array(list(class_dict.values()))
print(class_names)

test_images,test_label  = next(pred_gen)
#print(test_label)
#print(len(test_images))

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(test_images[:10])


#Effettuiamo la predizione
predictions = model.predict(pred_gen)

#Dato che test_label e un array a 2 dimensioni dove per ogni immagine abbiamo un array composto da 10 elementi, e la classe dell'immagine è indicata con la 
#posizione di un 1 in questo array.Abbiamo bisogno di una funzione che ci restituisca questa posizione partendo dall'array
def get_label(labels):
    i=0
    for l in labels:
        if(l==1):
            return i
        else:
            i=i+1
    print('Classe non trovata.FUNZIONE GET_LABEL')


#Costruiamo una funzione per graficare la predizione
def plot_image(i, predictions_array, true_label, img):
  predictions_array, True_label, img = predictions_array, get_label(true_label[i]), img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  #print(predicted_label)
  #print(True_label)
  if predicted_label == True_label: 
    color = 'blue'  #Se la predizione e' corretta coloriamo il nome di blue
  else:
    color = 'red'   #Altrimenti lo coloriamo di rosso

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[True_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, get_label(true_label[i])
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Effettuiamo la predizione del primo elemento di test
#sample_training_images, _ = next(pred_gen)

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)

#print(predictions[i])#TEST

plot_image(i, predictions[i], test_label, test_images)

plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_label)
plt.show()


#Mostriamo i risultati con 15 immagini
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_label, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_label)
plt.tight_layout()
plt.show()