# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:01:13 2018

@author: nsde
"""

#%% Packages
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Dense, Flatten
# from tensorflow.python.keras.layers import Conv2D, MaxPool2D
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers import Dense, Multiply, Add, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from ddtn.transformers.transformer_util import get_transformer_init_weights
from ddtn.transformers.transformer_util import get_transformer_dim
from keras import backend as K

def my_init(shape, dtype=None):
    weights = get_transformer_init_weights(50, transformer_name='affine')
    return weights[0]

def my_init_bias(shape, dtype=None):
    weights = get_transformer_init_weights(50, transformer_name='affine')
    return weights[1]

#model.add(Dense(64, kernel_initializer=my_init))

#%%
# def get_loc_net(input_shape, transformer_name = 'affine'):
#     """ Example on how a localization layer can look like """
#     # Get dimension for the last layer
#     dim = get_transformer_dim(transformer_name)
#
#     # TODO: find out why the zero weights destroy the affine_diffeo and CPAB
#     # Get weights for identity transformer. Note 50=#unit in second last layer
#     # weights = get_transformer_init_weights(50, transformer_name)
#
#     # Construct localization net
#     locnet = Sequential()
#     locnet.add(Conv2D(16, (3,3), activation='tanh', input_shape=input_shape))
#     locnet.add(MaxPool2D(pool_size=(2,2)))
#     locnet.add(Conv2D(32, (3,3), activation='tanh'))
#     locnet.add(MaxPool2D(pool_size=(2,2)))
#     locnet.add(Conv2D(32, (3,3), activation='tanh'))
#     locnet.add(MaxPool2D(pool_size=(2,2)))
#     locnet.add(Flatten())
#     locnet.add(Dense(50, activation='tanh'))
#     locnet.add(Dense(dim, activation='tanh'))
#     return locnet
def get_loc_net(input_shape, transformer_name='affine', identity=True):
    """ Example on how a localization layer can look like """
    # Get dimension for the last layer
    dim = get_transformer_dim(transformer_name)

    # TODO: find out why the zero weights destroy the affine_diffeo and CPAB
    # Get weights for identity transformer. Note 50=#unit in second last layer
    # weights = get_transformer_init_weights(50, transformer_name)

    # Construct localization net
    locnet = Sequential()
    locnet.add(Conv2D(16, (3, 3), activation='tanh', input_shape=input_shape))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(32, (3, 3), activation='tanh'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(32, (3, 3), activation='tanh'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Flatten())
    locnet.add(Dense(50, activation='tanh'))
    if identity:
        locnet.add(Dense(dim, activation='tanh', kernel_initializer=my_init, bias_initializer=my_init_bias))
    else:
        locnet.add(Dense(dim, activation='tanh'))
    return locnet

def get_loc_net_func(input_shape, prevlayer, transformer_name='affine'):
    """ Example on how a localization layer can look like """
    # Get dimension for the last layer
    dim = get_transformer_dim(transformer_name)

    # TODO: find out why the zero weights destroy the affine_diffeo and CPAB
    # Get weights for identity transformer. Note 50=#unit in second last layer
    # weights = get_transformer_init_weights(50, transformer_name)

    # Construct localization net
    #locnet = Sequential()
    conv = Conv2D(16, (3, 3), activation='tanh', input_shape=input_shape)(prevlayer)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = Conv2D(32, (3, 3), activation='tanh')(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = Conv2D(32, (3, 3), activation='tanh')(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = Flatten()(conv)
    dense = Dense(50, activation='tanh')(conv)
    dense = Dense(dim, activation='tanh')(dense)
    return dense

#%%
if __name__ == "__main__":
    loc_net = get_loc_net((250, 250, 1), transformer_name='affine')