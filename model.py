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

def inception_block(x,one_size,three_size_reduce, five_size_reduce, three_size, five_size,pool_project):

    x1 = tf.layers.Conv2D(filters=one_size, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)



    x2 = tf.layers.Conv2D(filters=three_size_reduce, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)    
    x5 = tf.layers.Conv2D(filters=three_size, kernel_size=(3,3), padding='SAME', activation = 'relu')(x2)



    x3 = tf.layers.Conv2D(filters=five_size_reduce, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)
    x6 = tf.layers.Conv2D(filters=five_size, kernel_size=(5,5), padding='SAME', activation = 'relu')(x3)



    x4 = tf.layers.MaxPooling2D(pool_size = (3,3), strides=(1,1), padding = 'SAME')(x)
    x7 = tf.layers.Conv2D(filters=pool_project, kernel_size=(1,1), padding='SAME', activation = 'relu')(x4)
    out = tf.layers.Concatenate()([x7, x6, x5, x1])
    return out

def getGoogleNet(numClass):
    inputs = tf.Input(shape=(224,224,3), name = 'img')
    x = tf.layers.Conv2D(filters = 64, kernel_size = (7,7), padding = 'SAME',
                            activation = 'relu')(inputs)
    x = tf.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)
    x = tf.layers.Conv2D(filters = 64, kernel_size = (7,7), padding = 'SAME',
                            activation = 'relu', strides = (1,1))(inputs)
    x = tf.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,64,96,16,128, 32,32)
    x = inception_block(x,128,128,32,192,96,64)
    x = tf.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,192,96,16,208,48,64)
    x = inception_block(x,160,112,24,224,64,64)
    x = inception_block(x,128,128,24,256,64,64)
    x = inception_block(x,112,144,32,288,64,64)
    x = inception_block(x,256,160,32,320,128,128)
    x = tf.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,256,160,32,320,128,128)
    x = inception_block(x,384,192,48,384,128,128)
    x = tf.layers.AveragePooling2D(pool_size = (7,7), strides=(1,1), padding = 'valid')(x)

    x = tf.layers.Dropout(0.4)(x)
    x = tf.layers.Dense(1000, activation = 'relu')(x)
    x = tf.layers.Dense(numClass, activation = 'softmax')(x)

    out= tf.Model(inputs, x, name='inception_v1')

    return out

def resNetBlock(x, filtters):
    x = tf.keras.layers.Conv2D(filters = filtters,kernel_size = (3,3),padding = 'same', activation = 'relu')(x)
    out = tf.keras.layers.Conv2D(filters = filtters,kernel_size = (3,3),padding = 'same', activation = 'relu')(x)   
    return out


def getResNetModel(numClass):
    inputs = tf.keras.input(shape=(224,224,3), name = 'img')
    x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,1), padding = 'valid')(inputs)

    xb = resNetBlock(x,64)
    x1 = tf.keras.layers.Concatenate()([x, xb])
    xb = resNetBlock(x1,64)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,64)
    x1 = tf.keras.layers.Concatenate()([x1, xb])

    xb = resNetBlock(x1,128)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,128)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,128)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,128)
    x1 = tf.keras.layers.Concatenate()([x1, xb])

    xb = resNetBlock(x1,256)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,256)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,256)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,256)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,256)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,256)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    
    xb = resNetBlock(x1,512)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,512)
    x1 = tf.keras.layers.Concatenate()([x1, xb])
    xb = resNetBlock(x1,512)
    x1 = tf.keras.layers.Concatenate()([x1, xb])


    xavg = tf.keras.layers.AveragePooling2D(pool_size = (3,3), strides = (1,1), padding = 'valid')(x1)
    x = tf.keras.layers.Dense(numClass, activation = 'softmax')(xavg)