import os
import glob

##PROVA DI ACCESSO E CREAZIONE DIZIONARIO DI PATH DI IMMAGINI
dataset_dir= 'C:\\Users\\stefr\\Desktop\\TESI DATASET\\PlantVillage'
classes=os.listdir(dataset_dir) #Recuperiamo la cartella contenente le classi
classes=sorted(classes)         #Ordinimo le classi in ordine alfabetico
print(type(classes))            #Scriviamo il tipo di classe
print(len(classes))             #Scriviamo quante classi ci sono
print(classes[0:10])            #Scriviamo tutte le classi


#Per gestire le classi creeremo un dizionario.Ogni classe sarà la chiave mentre gli elementi
#del dizionario saranno delle liste contenti i path delle immagini di una data classe.
#Per ottenere la lista delle immagini jpg contenuta in ogni sottocartella ci avvarremmo del modulo
#glob.


paths=dict()

for cl in classes:
    cl_paths=sorted(glob.glob(os.path.join(dataset_dir,cl,"*.jpg"))) #os.path ci permette di creare i path delle immagini contenute nella cartella "dataset_dir/cl/*.jpg" cioè selezioniamo tutte le immagini
    #cl_paths sarà una lista dei path delle immagini contenute nalla cartella "dataset_dir/cl"
    paths[cl]=cl_paths  #Salviamo cl_paths come elemento corrispondente alla classe nel dizionario

#testiamo il dizionario
print(paths["Tomato_healthy"][0:10])


