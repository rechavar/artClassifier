import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras as keras

def linealModel(num_class):
    model = keras.Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(num_class, activation='softmax'))
    return model

def LeNet(num_class):
    model = keras.Sequential()
    model.add(
        Conv2D(
            filters=6, kernel_size=(5,5), activation='tanh',
            padding='same', input_shape=(32, 32, 3)
        )
    )
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (5,5), activation='tanh'))
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(120, (5,5), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(num_class, activation='softmax'))
    return model

def AlexNet(num_class):
    model = keras.Sequential()
    model.add(
        Conv2D(
            filters=96, kernel_size=(11,11), activation='relu',
            padding='same', input_shape=(224, 224, 3),  strides=(4,4),
        )
    )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(
        Conv2D(
            filters=256, kernel_size=(5,5), activation='relu',
            padding='same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(
        Conv2D(
            filters=384, kernel_size=(5,5), activation='relu',
            padding='same'
        )
    )
    model.add(
        Conv2D(
            filters=384, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )    
    model.add(
        Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_class, activation='softmax'))
    return model
    