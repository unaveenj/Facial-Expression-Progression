from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
import pickle
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import TensorBoard
import keras
from sklearn.preprocessing import LabelEncoder
from random import shuffle


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tensorboard = TensorBoard(log_dir = '{}'.format('check'))
img_size = 224

train_data = np.load('Data.npy', allow_pickle=True)
shuffle(train_data)
x = np.array([i[0] for i in train_data]).reshape(-1,img_size,img_size,3)
y = [i[1] for i in train_data]


from tensorflow.keras.utils import to_categorical
le = LabelEncoder()
labels = le.fit_transform(y)
labels = to_categorical(labels)


print(len(x),len(y))
print(y[:10])
print(x.shape)
lb = LabelBinarizer()
#labels = lb.fit_transform(y)
data = np.array(x)
#labels = np.array(y)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.33, random_state=42)
#print(len(lb.classes_))


MODEL_NAME = 'model_evil_port.model'
tensorboard = TensorBoard(log_dir = 'logs'.format(MODEL_NAME))



trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
valAug = ImageDataGenerator()


def model_():
    conv_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = Flatten()(conv_model.output)
    # three hidden layers
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    
    predictions = Dense(len([i for i in os.listdir('Dataset/')]), activation='softmax')(x)
    
    full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

    for layer in conv_model.layers:
        layer.trainable = False

    full_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.0001),
                  metrics=['acc'])

    return full_model

full_model = model_()
#with augmentation

#history = full_model.fit_generator(trainAug.flow(trainX, trainY, batch_size=32),steps_per_epoch=len(trainX) // 32,validation_data=valAug.flow(testX, testY),validation_steps=len(testX) // 32,epochs=10,callbacks = [tensorboard]) 
#full_model.save(MODEL_NAME)

# with out augmentation

history = full_model.fit(trainX, trainY, batch_size=32, epochs=8 ,validation_data = (testX,testY),callbacks = [tensorboard])
full_model.save('Model_vgg16_tL.model')

import matplotlib.pyplot as plt
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
