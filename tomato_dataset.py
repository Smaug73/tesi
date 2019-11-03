#Questo modulo conterrà la classe che sarà importata per la creazione del dataset di immagini
# e per il suo utilizzo per il training e test della rete neurale.

import os
import glob
import numpy as np
from copy import copy
from skimage import io as sio
from matplotlib import pyplot as plt



class Dataset:
    #Costruttore della classe, costruisce se stesso partendo dal path della directory contenenti le classi
    def __init__(self ,path_to_dataset):

        self.path_to_dataset = path_to_dataset
        classes=os.listdir(path_to_dataset)
        self.paths=dict()
        #Con questo ciclo popoliamo il dizionario con i path di tutte le immagini di una relativa classe
        #I path sono raggruppati attraverso il nome della classe a cui appartengono
        for cl in classes: 
            current_paths= sorted(glob.glob(os.path.join(path_to_dataset,cl,"*.jpg")))
            self.paths[cl]= current_paths



    #metodo che ritorna il path di una immagine passando l'indice dell'immmagine
    def getImagePath(self,cl,idx):
        return self.paths[cl][idx]

    

    #Metodo che ritorna la lista di tutte le classi del dataset
    def getClasses(self):
        return sorted(self.paths.keys())


    #Metodo per mostrare una immagine
    def showImage(self,class_name,image_number):
        print(self.getImagePath(class_name,image_number))
        image = sio.imread(self.getImagePath(class_name,image_number))  #carichiamo l'immagine
        
        plt.figure()        #creaimo una figura
        plt.imshow(image)   #inseriamo l'immagine nella figura
        plt.show()            #mostriamo la figura



    #metodo per creare un sottoinsieme delle immagini presenti nelle classi del dataset
    #Utile per la creazione di un train set ed un test set.
    #Ritornerà due dataset, uno per il training e uno per il testing
    def splitTrainingTest(self,percent_train):
        
        #Creiamo i dizionari che conterranno i path di training e test.
        training_paths=dict()   
        test_paths=dict()         

        for cl in self.getClasses():
            paths=self.paths[cl]                                #otteniamo la lista dei path
            shuffled_paths = np.random.permutation(paths)       #randomizzaimo i path
            split_idx= int(len(shuffled_paths)*percent_train)   #calcoliamo la quantità di immagini che faranno parte del training set,arrotondando il numero facendo il cast int,il numero sarà anche l'indice limite del train set
            training_paths[cl]=shuffled_paths[0:split_idx]      #salva le prime "split_idx" immagini nel training set
            test_paths[cl]=shuffled_paths[split_idx::]          #assegniamo al test_set i path rimanenti, che sarebbero quelli che vanno dall'indice di split fino alla fine della lista.

            training_dataset = copy(self)
            training_dataset.paths=training_paths

            test_dataset= copy(self)
            test_dataset.paths = test_paths

        return training_dataset,test_dataset



    #Metodo per la creazione di array di immagini
    #Sono i metodi creano le liste da passare al modello
    def createArrayImgs(self):
        #img_array=np.array([])
        img_list=list()
        label_list= list()

        for cl in self.getClasses():
            imgs_paths= self.paths[cl]
            for s in imgs_paths:
                img_list.append(s)
                label_list.append(cl)

        return img_list,label_list