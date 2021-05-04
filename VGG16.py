import keras
from keras import backend as K
from keras.layers import Dense, Flatten, InputLayer
from keras.models import Sequential, Model
from keras.applications import VGG16


INPUT_SHAPE = (128, 128, 3)
 

def create_model():
    model = Sequential()
    model.add(InputLayer(INPUT_SHAPE))
    model.add(VGG16(weights='imagenet', include_top=False))
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))
    return model