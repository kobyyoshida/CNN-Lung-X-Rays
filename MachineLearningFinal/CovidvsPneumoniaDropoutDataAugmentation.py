# -*- coding: utf-8 -*-
"""
@author: Michael Lewson
"""

import os, shutil
print("Starting Up! FOR PRECISION RECALL")

#where the covid folder is located in the Curated X-Ray Dataset
covid_dataset_dir = '/Users/Micha/Downloads/lung-images/Curated X-Ray Dataset/COVID-19'
#where the healthy folder is located in the Curated X-Ray Dataset
pneumonia_dataset_dir = '/Users/Micha/Downloads/lung-images/Curated X-Ray Dataset/Pneumonia-Viral'

#the directory created with the training, testing, and validation folders
base_dir = '/Users/Micha/Downloads/lung-images/Curated X-Ray Dataset/CovidVsPneumoniaRecallPrecisionvF'
#directory used to create machine learning

os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_covid_dir = os.path.join(train_dir, 'covid')
os.mkdir(train_covid_dir)

train_healthy_dir = os.path.join(train_dir, 'pneumonia')
os.mkdir(train_healthy_dir)

validation_covid_dir = os.path.join(validation_dir, 'covid')
os.mkdir(validation_covid_dir)

validation_healthy_dir = os.path.join(validation_dir, 'pneumonia')
os.mkdir(validation_healthy_dir)

test_covid_dir = os.path.join(test_dir, 'covid')
os.mkdir(test_covid_dir)

test_healthy_dir = os.path.join(test_dir, 'pneumonia')
os.mkdir(test_healthy_dir)

#populates the directories
print("Populating Training, Validation, and Testing Directories")
fnames = ['COVID-19 ({}).jpg'.format(i) for i in range(1, 897)] #Training set of images for Covid-19
for fname in fnames:
    src = os.path.join(covid_dataset_dir, fname)
    dst = os.path.join(train_covid_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['COVID-19 ({}).jpg'.format(i) for i in range(897, 1025)] # Validation set of images for Covid-19
for fname in fnames:
    src = os.path.join(covid_dataset_dir, fname)
    dst = os.path.join(validation_covid_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['COVID-19 ({}).jpg'.format(i) for i in range(1025, 1281)] # Test set of images for Covid-19
for fname in fnames:
    src = os.path.join(covid_dataset_dir, fname)
    dst = os.path.join(test_covid_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['Pneumonia-Viral ({}).jpg'.format(i) for i in range(1, 897)] #Training set of images for pneumonia
for fname in fnames:
    src = os.path.join(pneumonia_dataset_dir, fname)
    dst = os.path.join(train_healthy_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['Pneumonia-Viral ({}).jpg'.format(i) for i in range(897, 1025)] #Validation set of images for pneumonia
for fname in fnames:
    src = os.path.join(pneumonia_dataset_dir, fname)
    dst = os.path.join(validation_healthy_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['Pneumonia-Viral ({}).jpg'.format(i) for i in range(1025, 1281)] #Testing set of images for pneumonia
for fname in fnames:
    src = os.path.join(pneumonia_dataset_dir, fname)
    dst = os.path.join(test_healthy_dir, fname)
    shutil.copyfile(src, dst)
    
print("Initiating Covnet")
#instantiating a small covnet for pneumonia vs covid classification
from keras import layers
from keras import models

#configuring model for training
from keras import optimizers
#using imagedatagenerator to read images from directories

from keras.preprocessing.image import ImageDataGenerator

#setting up data augmentation configuration via imagedatagenrator
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#defines covnet with dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


import keras 

from keras import backend as K

#to calculate precision, recall
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


metrics2 = [keras.metrics.TruePositives(name='tp'),
           keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'),
           keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.BinaryAccuracy(name='accuracy'),
           precision,
           recall,
           keras.metrics.AUC(name='auc')]

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=metrics2)

#training the covnet with data augmentation generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100, #switching to 30 from 100
      validation_data=validation_generator,
      validation_steps=50)

print("Saving Model!")
model.save('CovidVsPneumoniaPrecisionRecallvf1.h5')


#showing loss and accuracy curves again
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#graphs
import matplotlib.pyplot as plt
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy w/ Dropout')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss w/ Dropout')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
        test_dir, 
        target_size = (150, 150), 
        batch_size = 20, 
        class_mode = 'binary')

#Calculates out precision, val_precision, recall, val_recall
precision2 = history.history['precision']
recall2 = history.history['recall']
averagePrecision = 0
totalPrecision = 0
for i in range(0, len(precision2)):
    totalPrecision += precision2[i]

FinalPrecision = totalPrecision/len(precision2)

totalValPrecision = 0
valprecision = history.history['val_precision']
for i in range(0, len(valprecision)):
    totalValPrecision += valprecision[i]
FinalValPrecision = totalValPrecision/len(valprecision)

totalValRecall = 0
valRecall = history.history['val_recall']
for i in range(0, len(valRecall)):
    totalValRecall += valRecall[i]
FinalValRecall = totalValRecall/len(valRecall)

totalRecall = 0
for i in range(0, len(recall2)):
    totalRecall += recall2[i]

FinalRecall = totalRecall/len(recall2)

totalAccuracy = 0
accuracy2= history.history['accuracy']
for i in range(0, len(accuracy2)):
    totalAccuracy += accuracy2[i]
FinalAccuracy = totalAccuracy/len(accuracy2)

totalValAccuracy = 0
valAccuracy = history.history['val_accuracy']
for i in range(0, len(valAccuracy)):
    totalValAccuracy += valAccuracy[i]
FinalValAccuracy = totalValAccuracy/len(valAccuracy)

#Prints out information
print("\n")

print("accuracy:", FinalAccuracy)
print("val_accuracy:", FinalValAccuracy)
print("precision:", FinalPrecision)
print("val_precision:", FinalValPrecision)
print("recall:", FinalRecall)
print("val_recall:", FinalValRecall)

print("\nFINISHED!")