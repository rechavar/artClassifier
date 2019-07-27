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

def inception_block(x,one_size, three_size, five_size):
    x1 = tf.keras.layers.Conv2d(filters=one_size, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)
    x2 = tf.keras.layers.Conv2d(filters=one_size, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)
    x3 = tf.keras.layers.Conv2d(filters=one_size, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)
    x4 = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)
    x5 = tf.keras.layers.Conv2d(filters=three_size, kernel_size=(3,3), padding='SAME', activation = 'relu')(x2)
    x6 = tf.keras.layers.Conv2d(filters=five_size, kernel_size=(5,5), padding='SAME', activation = 'relu')(x3)
    x7 = tf.keras.layers.Conv2d(filters=one_size, kernel_size=(1,1), padding='SAME', activation = 'relu')(x4)
    out = tf.keras.layers.Concatenate()([x7, x6, x5, x1])
    return out

def getGoogleNet(numClass):
    inputs = keras.input(shape=(224,224,3), name = 'img')
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7,7), padding = 'SAME',
                            activation = 'relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7,7), padding = 'SAME',
                            activation = 'relu', strides = (1,1))(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,64,96,16)
    x = inception_block(x,128,128,32)
    x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,192,96,16)
    x = inception_block(x,160,112,24)
    x = inception_block(x,128,128,24)
    x = inception_block(x,112,144,32)
    x = inception_block(x,256,160,32)
    x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,256,160,32)
    x = inception_block(x,384,192,48)
    x = tf.keras.layers.AveragePooling2D(pool_size = (7,7), strides=(1,1), padding = 'valid')(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1000, activation = 'relu')(x)
    x = tf.keras.layers.Dense(9, activation = 'softmax')(x)

    out= tf.keras.Model(input_layer, x, name='inception_v1')

    return out

