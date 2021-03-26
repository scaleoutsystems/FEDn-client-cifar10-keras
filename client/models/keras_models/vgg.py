'''VGG11/13/16/19 in keras.'''
import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from keras.layers import Dense, BatchNormalization, Flatten, Input
from keras.models import Sequential
from keras import activations


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def VGG(input_shape=(32,32,3), dimension='VGG16'):

    num_classes = 10

    model = Sequential()
    model.add(keras.Input(shape=input_shape))
    for x in cfg[dimension]:
        if x == 'M':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            print("x: ", x)
            model.add(Conv2D(x, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(activations.relu))
    #model.add(Flatten())
    model.add(AveragePooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model



