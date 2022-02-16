# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:10:17 2021

@author: nikaj
"""

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# load the dataset 
from keras.datasets import cifar10
# setting class names 
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#loading the dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#checking no of rows and columns 
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# view the data 
plt.imshow(x_train[0])

# checking unique of classes 
print(np.unique(y_train))
print(np.unique(y_test))

# standardising the images
x_train = x_train/255.0
x_test = x_test /255.0

# creating a sequential model and adding layers 
model = Sequential()

# first layer
model.add(Convolution2D(filters=32, kernel_size=3, activation = "relu", input_shape=[32,32,3]))
# second layer 
model.add(Convolution2D(filters= 32, kernel_size=3, activation= "relu"))
# maxpooling layer 
model.add(MaxPool2D(pool_size= 2,strides= 2))
# third layer
model.add(Convolution2D(filters=64,kernel_size=3,activation="relu"))
# fourth layer 
model.add(Convolution2D(filters=64,kernel_size=3,activation="relu"))
# max pooling layer 
model.add(MaxPool2D(pool_size= 2,strides= 2))
# flattern layer 
model.add(Flatten())

# adding fully connected layer
model.add(Dense(units= 20, activation = "relu"))
# output layer
model.add(Dense(units=10,activation= "softmax" ))

model.summary()

# compliling the model 
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", 
              metrics=["sparse_categorical_accuracy"])

# training the model 
history = model.fit(x_train,y_train,epochs=15,validation_data=(x_test,y_test))

# predictions 
pred = model.predict(x_test)
pred


# Accuracy curve
plt.figure(figsize=[6,4])
plt.plot(history.history['sparse_categorical_accuracy'], 'black')
plt.plot(history.history['val_sparse_categorical_accuracy'], 'blue')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')

# loss curve 
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black')
plt.plot(history.history['val_loss'], 'blue')
plt.legend(['Training loss', 'Validation loss'])
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Loss Curves')


