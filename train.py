import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv2D, MaxPooling2D, Average, Concatenate, Add, Activation
from tensorflow.keras.layers import Flatten,AveragePooling2D,AveragePooling2D,SeparableConv2D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from keras.models import Model,load_model
from tensorflow.keras import layers 
from tensorflow.keras.layers import RandomRotation,RandomFlip,RandomZoom,RandomContrast,RandomBrightness,RandomTranslation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras import activations
from keras import regularizers
import math


train_datagen= ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split = 0.2,
    rescale=1./255
)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2) 


train_images = train_datagen.flow_from_directory(
    "/kaggle/input/insects/New Dataset",
    target_size=(224,224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle = True,
    seed=42,
    subset = 'training',
)


val_images = datagen_val.flow_from_directory(
   "//kaggle/input/insects/New Dataset",
    target_size=(224,224),
    seed = 42,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    subset = 'validation'
)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

# Define the learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
def inception_module(layer_in, f1, f2, f3, f4):

    branch1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    
    branch2 = Conv2D(f2, (1,1), padding='same', activation='relu')(layer_in)
    branch21 = Conv2D(f2, (3,1), padding='same', activation='relu')(branch2)
    branch22 = Conv2D(f2, (1,3), padding='same', activation='relu')(branch2)
    
    branch3 = Conv2D(f2, (1,1), padding='same', activation='relu')(layer_in)
    branch3 = Conv2D(f3, (3,3), padding='same', activation='relu')(branch3)
    branch31 = Conv2D(f3, (3,1), padding='same', activation='relu')(branch3)
    branch32 = Conv2D(f3, (1,3), padding='same', activation='relu')(branch3)
    
    pool = AveragePooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool = Conv2D(f4, (1,1), padding='same', activation='relu')(pool)
    
    x = Concatenate(axis=3)([branch1, branch21, branch22, branch31, branch32, pool])
    
    return x


def reduction(layer_in, f1,f2):
    branch1 = Conv2D(f1,(1,1),padding = 'same', activation = 'relu')(layer_in)
    branch1 = Conv2D(f1, (3,3), padding = 'same', activation = 'relu')(branch1)
    branch1 = Conv2D(f1,(3,3),padding = 'same', activation = 'relu')(branch1)
    
    branch2 = Conv2D(f2,(3,3),padding = 'same', activation = 'relu')(layer_in)
    
    branch3 = MaxPooling2D((3,3), strides = (1,1), padding = 'same')(layer_in)

    x = Concatenate(axis = 3)([branch1,branch2,branch3])
    
    return x


def resnet(layer_in, f1,f2,f3):
    branch1 = Conv2D(f1,(1,1),padding = 'same', activation = 'relu')(layer_in)
    branch1 = Conv2D(f1,(1,3),padding = 'same', activation = 'relu')(branch1)
    branch1 = Conv2D(f1,(3,1), padding = 'same', activation = 'relu')(branch1)
    
    branch2 = Conv2D(f2,(1,1), padding = 'same', activation = 'relu')(layer_in)
    
    combine = Concatenate(axis = 3)([branch1,branch2])
    combine = Conv2D(f3,(1,1),padding = 'same',activation = 'relu')(combine)
    
    x = Add()([combine,layer_in])
    
    return x


def get_model() -> Model:
    input_shape = [224,224,3]
    inputs = tf.keras.Input(shape=input_shape)
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.2)(x)
    x = RandomZoom(0.1)(x)
    x = RandomContrast(0.1)(x)
    #x = RandomBrightness(0.2)(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid', strides = (2,2))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid', strides = (2,2))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size = 3 , strides=2)(x)
    x = inception_module(x ,64, 96, 128, 16)
    x = resnet(x,64,96,528)
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)
    x = inception_module(x ,64, 96, 128, 16)
    x = reduction(x, 256, 160)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)
    b = Dropout(0.25)(x)
    b = Flatten()(b)
    b = Dense(units = 500, activation = 'relu')(b)
    b = Dropout(0.4)(b)
    b = Dense(units = 100, activation = 'relu')(b)
    b = Dropout(0.4)(b)
    output = Dense(units = 27,activation='softmax')(b)


    model_custom = Model(inputs=inputs, outputs=output)
    model_custom.compile(optimizer = Adam(learning_rate=0.0001),loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])
    model_custom.build([ None, 224, 224, 3])
    return model_custom

model_custom = get_model()
model_custom.summary()

from tensorflow.keras.utils import plot_model
plot_model(model_custom,to_file='model_plot33.png',show_layer_activations=True)

history = model_custom.fit(train_images, validation_data=val_images, epochs=100, callbacks=[reduce_lr, early_stopping])

plt.figure(figsize=(15, 10))
plt.plot(history_custom.history['loss'], label='Training Loss')
plt.plot(history_custom.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=16)
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15, 10))
plt.plot(history_custom.history['accuracy'], label='Training Accuracy')
plt.plot(history_custom.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training and Validation Accuracy', fontsize=16)
plt.legend()
plt.grid()
plt.show()

model_custom.save('model_custom.keras')

