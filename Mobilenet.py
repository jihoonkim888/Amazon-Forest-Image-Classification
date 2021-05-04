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
from tf.keras.applications import MobileNet

# Define MobileNet model for Haze removal
def create_model():
    img = Input(shape = (128, 128, 3))
    model_mob = MobileNet(include_top=False, weights='imagenet', input_tensor=img, input_shape=None, pooling='avg')

    final_layer = model_mob.layers[-1].output
    dense_layer_1 = Dense(128, activation = 'relu')(final_layer)
    output_layer = Dense(17, activation = 'sigmoid')(dense_layer_1)

    model = Model(model_mob.input, output_layer)

    return model
