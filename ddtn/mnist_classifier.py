# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:24:41 2018

@author: nsde
"""

#%% Packages
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, InputLayer, MaxPool2D, Flatten, Lambda

from ddtn.transformers.construct_localization_net import get_loc_net
from ddtn.transformers.transformer_util import get_keras_layer
from ddtn.data.mnist_getter import get_mnist_distorted
from ddtn.helper.training_logger import KerasTrainingLogger

import argparse

#%% Argument parser for comman line input
def _argument_parser():
    parser = argparse.ArgumentParser(description='''This program will train a 
                                     neural network mnist dataset.''')
    parser.add_argument('-lr', action="store", dest="learning_rate", type=float, default = 0.0001,
                        help = '''Learning rate for optimizer. Default: 1e-4''')
    parser.add_argument('-ne', action="store", dest="num_epochs", type=int, default = 10,
                        help = '''Number of epochs. Default: 10''')
    parser.add_argument('-bs', action="store", dest="batch_size", type=int, default = 100,
                        help = '''Batch size: Default: 100''')
    parser.add_argument('-tt', action="store", dest="transformer_type", type=str, 
                        default='affine', help = '''Transformer type to use. 
                        Choose between: no, affine, cpab, affine_diffeo, homografy
                        or TPS''')
    res = parser.parse_args()
    args = vars(res)
    
    # Print input
    print(50*'-')
    print('Running script with arguments:')
    print('   learning rate:            ', args['learning_rate'])
    print('   number of epochs:         ', args['num_epochs'])
    print('   batch size:               ', args['batch_size'])
    print('   transformer type:         ', args['transformer_type'])
    print(50*'-')
    
    return args

#%%
if __name__ == '__main__':
    # Get input arguments
    args = _argument_parser()
    
    # Get some data
    X_train, y_train, X_test, y_test = get_mnist_distorted()
    
    # Data format
    input_shape=(60,60,1)
        
    # Keras model
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    
    if args['transformer_type'] != 'no': # only construct if we want to use transformers
        # Construct localization network
        loc_net = get_loc_net(input_shape=input_shape,
                              transformer_name=args['transformer_type'])
    
        # Add localization network and transformer layer
        transformer_layer = get_keras_layer(args['transformer_type'])
        model.add(transformer_layer(localization_net=loc_net, 
                                    output_size=input_shape))
    else:
        model.add(Lambda(lambda x: x)) # identity layer -> same model structure
   
    # Construct feature extraction network
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Get summary of model
    print('Global model')
    model.summary()
    
    # Training logger
    direc = './logs2'
    logger = KerasTrainingLogger(data=X_test[:10],
                                 name=args['transformer_type'],
                                 log_dir=direc, 
                                 trans_layer=0)
    
    # Fit model
    model.fit(X_train, y_train,
              batch_size=args['batch_size'],
              epochs=args['num_epochs'],
              validation_data=(X_test, y_test),
              callbacks=[logger])    

    # Save weights
    model.save_weights(direc + '/' + args['transformer_type'] + '/weights')
    
    