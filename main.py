import imageio
import glob
import numpy as np
import os
import csv
import tensorflow
from tensorflow.keras import layers, models

#Functie pentru normalizarea datelor
def normalizeImages(images):
     mean_image = np.mean(images, axis = 0)
     sd = np.std(images)
     return (images - mean_image) / sd

#Citirea datelor de antrenare
trainImages = []
for imagePath in glob.glob('C:/Users/*.png'):
     image = imageio.imread(imagePath)
     trainImages.append(image)
trainImages = np.array(trainImages)
trainImages = np.expand_dims(trainImages, axis = -1)

f = open('C:/Users/train.txt')
trainLabels = f.readlines()
for i in range(len(trainLabels)):
     trainLabels[i] = int(trainLabels[i][11])
trainLabels = np.array(trainLabels)

#Citirea datelor de testare
testImages = []
for imagePath in glob.glob('C:/Users/*.png'):
     image = imageio.imread(imagePath)
     testImages.append(image)
testImages = np.array(testImages)
testImages = np.expand_dims(testImages, axis = -1)

#Normalizarea datelor
trainImages = normalizeImages(trainImages)

testImages = normalizeImages(testImages)

#CNN
#Crearea modelului
model = models.Sequential()

#Adaugare layers
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

#Aplatizarea input-ului
model.add(layers.Flatten())

#Output
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(9))

#Compilare
model.compile(optimizer = 'adam', loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['acc'])

#Antrenare
model.fit(trainImages, trainLabels, epochs = 7)

#Probabilitatile de predictie
predictionResult = model.predict(testImages)

#Citesc numele pozelor de testare
def readNames():
     fileList = os.listdir('C:/Users/test')

     return fileList

a = readNames()

#Afisarea predictiilor
with open('output.csv', 'w') as file:
     writer = csv.writer(file)
     writer.writerow(['id', 'label'])

     for i in range(len(a)):
          writer.writerow([a[i], np.argmax(predictionResult[i], axis=-1)])
