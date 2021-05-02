import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import clear_session

import keras
from keras import backend as K
#from keras.regularizers import l2
#from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
#from keras.optimizers import Adam
from keras.applications import MobileNet, VGG16
#from tensorflow.keras.applications.vgg16 import VGG16

def create_model():
    model = Sequential()
    model.add(InputLayer(INPUT_SHAPE))
    model.add(VGG16(weights='imagenet', include_top=False))
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))
    return model