import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv2D, MaxPooling2D, Average
from tensorflow.keras.layers import Flatten,AveragePooling2D,AveragePooling2D,Activation,Attention,Multiply,SeparableConv2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RandomRotation,RandomFlip,RandomZoom,RandomContrast
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from tensorflow.keras import regularizers
import streamlit as st



@st.cache_resource
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
