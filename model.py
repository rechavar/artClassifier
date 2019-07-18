import tensorflow.keras as tf
from tensorflow.keras import layers

def getLinealModel(numClass):
    model = tf.Sequential([
        layers.Flatten(input_shape =(32,32,3)),
        layers.Dense(numClass,activation = 'softmax')
    ])
    return model

def getLetnetModel(numClass):
    model = tf.Sequential([
        layers.Conv2D(
            filters = 6, kernel_size =(5,5), activation = 'tanh', 
            padding = 'same', input_shape = (32,32,3)
        ),
        layers.AveragePooling2D(pool_size = (2,2)),
        layers.Conv2D(16,(5,5), activation = 'tanh'),
        layers.AveragePooling2D((2,2)),
        layers.Conv2D(16,(5,5), activation = 'tanh'),
        layers.Flatten(),
        layers.Dense(84, activation = 'tanh'),
        layers.Dense(numClass, activation = 'softmax')
    ])
    return model


def getAlexnetModel(numClass):
    model = tf.Sequential([
        layers.Conv2D(
            filters=96, kernel_size=(11,11), activation='relu',
            strides=(4,4), padding='valid', input_shape=(224, 224, 3)
        ),
        layers.MaxPooling2D(pool_size = (2,2), strides=(2,2), padding = 'valid'),
        layers.Conv2D(
            filters= 256, kernel_size = (11,11), activation = 'relu',
            strides = (1,1), padding = 'valid'
            ),
        layers.AveragePooling2D(pool_size = (2,2), strides=(2,2), padding = 'valid'),
        layers.Conv2D(
            filters= 384, kernel_size = (3,3), activation = 'relu',
            strides = (1,1), padding = 'valid'
            ),
        layers.Conv2D(
            filters= 384, kernel_size = (3,3), activation = 'relu',
            strides = (1,1), padding = 'valid'
            ),
        layers.Conv2D(
            filters= 256, kernel_size = (3,3), activation = 'relu',
            strides = (1,1), padding = 'valid'
            ),
        layers.AveragePooling2D(pool_size = (2,2), strides=(2,2), padding = 'valid'),
        layers.Flatten(),
        layers.Dense(4096, activation = 'relu'),
        layers.Dropout(0.4),
        layers.Dense(4096, activation = 'relu'),
        layers.Dropout(0.4),
        layers.Dense(1000, activation = 'relu'),
        layers.Dropout(0.4),
        layers.Dense(numClass,activation = 'softmax')
        
    ])
    return model

def getVGGModel(numClass):
    model = tf.Sequential([
        layers.Conv2D(
            filters = 64, kernel_size = (3,3), activation = 'relu',
            padding = 'same', input_shape = (224,224,3)
        ),
        layers.Conv2D(
            filters = 64, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
        layers.Conv2D(
            filters = 128, kernel_size = (3,3), activation = 'relu',
            padding = 'same' 
        ),
        layers.Conv2D(
            filters = 128, kernel_size = (3,3), activation = 'relu',
            padding = 'same' 
        ),
        layers.AveragePooling2D(pool_size = (2,2), strides = (2,2)),
        layers.Conv2D(
            filters = 256, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.Conv2D(
            filters = 256, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.AveragePooling2D(pool_size = (2,2), strides = (2,2)),
        layers.Conv2D(
            filters = 512, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.Conv2D(
            filters = 512, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.Conv2D(
            filters = 512, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.AveragePooling2D(pool_size = (2,2), strides = (2,2)),
        layers.Conv2D(
            filters = 512, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.Conv2D(
            filters = 512, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.Conv2D(
            filters = 512, kernel_size = (3,3), activation = 'relu',
            padding = 'same'
        ),
        layers.AveragePooling2D(pool_size = (2,2), strides = (2,2)),
        layers.Flatten(),
        layers.Dense(4096, activation = 'relu'),
        layers.Dense(4096, activation = 'relu'),
        layers.Dense(numClass, activation = 'softmax')])
    return model 
