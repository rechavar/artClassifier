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
    
def VGG16(num_class):
    model = tf.keras.Sequential()
    model.add(
        Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same', input_shape=(224, 224, 3)
        )
    )
    model.add(
        Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(
        Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    return model

def inception_block(x,one_size,three_size_reduce, five_size_reduce, three_size, five_size,pool_project):

    x1 = Conv2D(filters=one_size, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)



    x2 = Conv2D(filters=three_size_reduce, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)    
    x5 = Conv2D(filters=three_size, kernel_size=(3,3), padding='SAME', activation = 'relu')(x2)



    x3 = Conv2D(filters=five_size_reduce, kernel_size=(1,1), padding='SAME', activation = 'relu')(x)
    x6 = Conv2D(filters=five_size, kernel_size=(5,5), padding='SAME', activation = 'relu')(x3)



    x4 = MaxPooling2D(pool_size = (3,3), strides=(1,1), padding = 'SAME')(x)
    x7 = Conv2D(filters=pool_project, kernel_size=(1,1), padding='SAME', activation = 'relu')(x4)
    out = Concatenate()([x7, x6, x5, x1])
    return out

def InceptionV1(numClass):
    inputs = tf.Input(shape=(224,224,3), name = 'img')
    x = Conv2D(filters = 64, kernel_size = (7,7), padding = 'SAME',
                            activation = 'relu')(inputs)
    x = MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)
    x = Conv2D(filters = 64, kernel_size = (7,7), padding = 'SAME',
                            activation = 'relu', strides = (1,1))(inputs)
    x = MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,64,96,16,128, 32,32)
    x = inception_block(x,128,128,32,192,96,64)
    x = MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,192,96,16,208,48,64)
    x = inception_block(x,160,112,24,224,64,64)
    x = inception_block(x,128,128,24,256,64,64)
    x = inception_block(x,112,144,32,288,64,64)
    x = inception_block(x,256,160,32,320,128,128)
    x = MaxPooling2D(pool_size = (3,3), strides=(2,2), padding = 'valid')(x)

    x = inception_block(x,256,160,32,320,128,128)
    x = inception_block(x,384,192,48,384,128,128)
    x = AveragePooling2D(pool_size = (7,7), strides=(1,1), padding = 'valid')(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation = 'relu')(x)
    x = Dense(numClass, activation = 'softmax')(x)

    out= tf.Model(inputs, x, name='inception_v1')

    return out

def resnet_block(x, filtters):
    x = Conv2D(filters = filtters,kernel_size = (3,3),padding = 'same', activation = 'relu')(x)
    out = Conv2D(filters = filtters,kernel_size = (3,3),padding = 'same', activation = 'relu')(x)   
    return out


def getResNetModel(numClass):
    inputs = keras.input(shape=(224,224,3), name = 'img')
    x = MaxPooling2D(pool_size = (3,3), strides = (1,1), padding = 'valid')(inputs)

    xb = resnet_block(x,64)
    x1 = Concatenate()([x, xb])
    xb = resnet_block(x1,64)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,64)
    x1 = Concatenate()([x1, xb])

    xb = resnet_block(x1,128)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,128)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,128)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,128)
    x1 = Concatenate()([x1, xb])

    xb = resnet_block(x1,256)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,256)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,256)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,256)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,256)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,256)
    x1 = Concatenate()([x1, xb])
    
    xb = resnet_block(x1,512)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,512)
    x1 = Concatenate()([x1, xb])
    xb = resnet_block(x1,512)
    x1 = Concatenate()([x1, xb])


    xavg = AveragePooling2D(pool_size = (3,3), strides = (1,1), padding = 'valid')(x1)
    x = Dense(numClass, activation = 'softmax')(xavg)