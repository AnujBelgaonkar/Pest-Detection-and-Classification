import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv2D, MaxPooling2D, Average
from tensorflow.keras.layers import Flatten,AveragePooling2D,AveragePooling2D,Activation,Attention,Multiply,SeparableConv2D
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import RandomRotation,RandomFlip,RandomZoom,RandomContrast
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow.keras import regularizers


train_datagen= ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
   validation_split = 0.2,
    rescale=1./255
)

train_images = train_datagen.flow_from_directory(
    "/kaggle/input/agricultural-pests-image-dataset",
    target_size=(224,224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    seed=42,
    subset = 'training',
)


val_images = train_datagen.flow_from_directory(
   "/kaggle/input/agricultural-pests-image-dataset",
    target_size=(224,224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset = 'validation'
)

checkpoint_path = "custom_codel.weights.h5"
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor="val_accuracy", save_best_only=True)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

augment = Sequential([
    RandomFlip("horizontal"),            # Random horizontal flipping
    RandomRotation(factor=0.2),          # Random rotation up to 20 degrees
    RandomZoom(height_factor=0.1, width_factor=0.1),  # Random zooming up to 10%
    RandomContrast(factor=0.1),          # Random contrast adjustment up to 10%
])

def get_model() -> Model:
    input_shape = [224,224,3]
    inputs = tf.keras.Input(shape=input_shape)
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.2)(x)
    x = RandomZoom(0.1)(x)
    x = RandomContrast(0.1)(x)
    x = Conv2D(filters = 64, kernel_size = 2, activation = 'tanh')(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)
    x = Conv2D(filters = 128, kernel_size = 2, activation = 'tanh')(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)
    y = Conv2D(filters = 256, kernel_size = 2, activation = 'relu')(x)
    y = MaxPooling2D(pool_size = 2, strides = 2)(y)
    y = MaxPooling2D(pool_size = 2, strides = 2)(y)
    y = BatchNormalization()(y)
    z = Conv2D(filters = 256, kernel_size = 2, activation = 'relu')(x)
    z = MaxPooling2D(pool_size = 2, strides = 2)(z)
    z = MaxPooling2D(pool_size = 2, strides = 2)(z)
    z = BatchNormalization()(z)
    a = tf.keras.layers.Multiply()([y, z])
    b = Flatten()(a)
    b = Dense(units = 1000, activation = 'tanh',kernel_regularizer=regularizers.l2(0.001))(b)
    b = Activation(activations.relu)(b)
    b = Dropout(0.7)(b)
    b = Dense(units = 200, activation = 'tanh',kernel_regularizer=regularizers.l2(0.001))(b)
    b = Activation(activations.relu)(b)
    b = Dropout(0.7)(b)
    output = Dense(units = 12,activation='softmax')(b)


    model_custom = Model(inputs=inputs, outputs=output)
    model_custom.compile(optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),loss = 'categorical_crossentropy',
                         metrics = ['accuracy',keras.metrics.Recall(),keras.metrics.Precision()])
    model_custom.build([ None, 224, 224, 3])
    return model_custom

model_custom = get_model()
model_custom.summary()

from tensorflow.keras.utils import plot_model
plot_model(model_custom,show_layer_activations=True)

history_custom = model_custom.fit( train_images,
                    validation_data=val_images,
                    epochs = 110,
                    callbacks=[checkpoint_callback])

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

model.save_weights('model.weights.h5')

