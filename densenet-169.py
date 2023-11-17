import keras
import keras.utils
from keras import utils as np_utils

import tensorflow.keras.layers
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
import cv2
import numpy as np

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


train_datagen_with_aug = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen_with_aug.flow_from_directory(
    r'D:\SRM\Experience and Certificates\projects\bengali sign language\archive\RESIZED_DATASET',
    target_size=(224, 224),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical',
    subset='training'
    ) # set as training data

validation_generator = train_datagen_with_aug.flow_from_directory(
    r'D:\SRM\Experience and Certificates\projects\bengali sign language\archive\RESIZED_DATASET', # same directory as training data
    target_size=(224, 224),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
    ) # set as validation data

input_img = Input(shape=(224, 224, 3))

import keras
from keras.layers import Dense, Flatten
from keras.optimizers import SGD, Adam

from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.applications
from keras.applications import densenet

DENSENET_169 = Sequential()

densenet169_model = keras.applications.densenet.DenseNet169(
    include_top=False, weights='imagenet',
    input_shape=(224,224,3), pooling='max', classes=38
)

for layer in densenet169_model.layers:
    layer.trainable = False

DENSENET_169.add(densenet169_model)
DENSENET_169.add(Flatten())
DENSENET_169.add(Dropout(0.2))
DENSENET_169.add(Dense(1024,activation='relu'))
DENSENET_169.add(Dropout(0.2))
DENSENET_169.add(Dense(512,activation='relu'))
DENSENET_169.add(Dense(38, activation='softmax'))

DENSENET_169.summary()

adam = keras.optimizers.Adam(learning_rate=0.001)
#SGD = keras.optimizers.SGD(learning_rate=0.0001)
DENSENET_169.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy',precision_m, recall_m, f1_m])

graph = DENSENET_169.fit(train_generator, epochs=25, validation_data = validation_generator)
graph
DENSENET_169.save(r'D:\SRM\Experience and Certificates\projects\bengali sign language\archive\RESIZED_DATASET169')


import matplotlib.pyplot as plt

fig1 = plt.gcf()

plt.plot(graph.history['accuracy'])
plt.plot(graph.history['val_accuracy'])

# plt.axis(ymin=0.4, ymax=1)
plt.grid()

plt.title('DenseNet169 Model Accuracy for DR Dataset')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','validation'])

plt.show()

max_acc = max(graph.history['val_accuracy'])

print('The highest accuracy achieved using DenseNet169 model is',max_acc*100)
