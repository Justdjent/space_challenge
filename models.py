from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.layers import Input as Input_from_layers
# from tensorflow.python.keras.layers import Input as Input_from_layers

from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D, Dropout
from keras.layers.merge import concatenate
from keras.layers import Dense, Multiply, Add, Lambda, Flatten, Dot
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import keras.backend as K
from keras.layers import multiply
# from tensorflow.python.keras.models import Sequential
from keras.models import Sequential
# from inception_resnet_v2 import InceptionResNetV2
# from mobile_net_fixed import MobileNet
from ddtn.transformers.construct_localization_net import get_loc_net, get_loc_net_func
from ddtn.transformers.transformer_util import get_keras_layer
from ddtn.transformers.keras_layers import Round
from ddtn.data.mnist_getter import get_mnist_distorted
from ddtn.helper.training_logger import KerasTrainingLogger

#from tensorflow.python.keras.layers import InputLayer #  Dense, Conv2D,


from resnet50_fixed import ResNet50, conv_block, identity_block
from params import args
from sel_models.unets import (create_pyramid_features, conv_relu, prediction_fpn_block, conv_bn_relu, decoder_block_no_bn)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# set_session(tf.Session(config=config))

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def transformation_branch(weights, prevlayer, prefix):
    round_out = Round()(weights)
    transformed_image_1 = transformer_block(prevlayer)
    transformed_image_2 = transformer_block(prevlayer)
    transformed_image_3 = transformer_block(prevlayer)

    round_out_1 = Lambda(lambda x: x[:, 0])(round_out)
    round_out_2 = Lambda(lambda x: x[:, 1])(round_out)
    round_out_3 = Lambda(lambda x: x[:, 2])(round_out)
    transformed_image_1 = Multiply(name=prefix + "_mul1")([round_out_1, transformed_image_1])
    transformed_image_2 = Multiply(name=prefix + "_mul2")([round_out_2, transformed_image_2])
    transformed_image_3 = Multiply(name=prefix + "_mul3")([round_out_3, transformed_image_3])

    x = Add(name=prefix + "add")([transformed_image_1, transformed_image_2, transformed_image_3])
    return x

def classification_branch(prevlayer, prefix, out_number):

    # conv1 = conv_block_simple(prevlayer, 128, prefix + 'conv1')
    #conv2 = conv_block_simple(conv1, 192, prefix + 'conv2')
    # conv3 = Conv2D(10, (1, 1), padding="same", kernel_initializer="he_normal", activation='relu',
    #                strides=(3, 3), name=prefix + "_conv")(conv1)
    # x = BatchNormalization(axis=1, name=prefix + 'bn_conv1')(prevlayer)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = BatchNormalization(axis=1, name=prefix + 'bn_conv1')(prevlayer)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = csse_block(x, prefix + "csse_classification_1")
    x = conv_block(x, 3, [64, 64, 256], stage=2, block=prefix + 'a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=prefix + 'b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=prefix +'c')
    x = csse_block(x, prefix + "csse_classification_2")
    conv_red = Conv2D(10, (1, 1), padding="same", kernel_initializer="he_normal", activation='relu',
                   strides=(3, 3), name=prefix + "_conv")(x)
    pool = AveragePooling2D(pool_size=(16, 16), padding='valid')(conv_red)
    flat = Flatten()(pool)
    lin1 = Dense(128, name=prefix + '_dense1', activation='relu')(flat)
    lin2 = Dense(64, name=prefix + '_dense2', activation='relu')(lin1)
    drop = Dropout(0.5)(lin2)
    result = Dense(out_number, name=prefix + '_output', activation='softmax')(drop)
    #x = Multiply()([prevlayer, lin2])
    return result


def transformation_branch_3(weights, prevlayer, prefix):
    round_out = Round()(weights)
    transformed_image_1 = transformer_block(prevlayer)
    transformed_image_2 = transformer_block(prevlayer)
    transformed_image_3 = transformer_block(prevlayer)

    round_out_1 = Lambda(lambda x: x[:, 0])(round_out)
    round_out_2 = Lambda(lambda x: x[:, 1])(round_out)
    round_out_3 = Lambda(lambda x: x[:, 2])(round_out)
    transformed_image_1 = Multiply(name=prefix + "_mul1")([round_out_1, transformed_image_1])
    transformed_image_2 = Multiply(name=prefix + "_mul2")([round_out_2, transformed_image_2])
    transformed_image_3 = Multiply(name=prefix + "_mul3")([round_out_3, transformed_image_3])

    x = Add(name=prefix + "add")([transformed_image_1, transformed_image_2, transformed_image_3])
    return x

def transformation_branch_5(weights, prevlayer, prefix):
    round_out = Round()(weights)
    transformed_image_1 = transformer_block(prevlayer)
    transformed_image_2 = transformer_block(prevlayer)
    transformed_image_3 = transformer_block(prevlayer)
    transformed_image_4 = transformer_block(prevlayer)
    transformed_image_5 = transformer_block(prevlayer)


    round_out_1 = Lambda(lambda x: x[:, 0])(round_out)
    round_out_2 = Lambda(lambda x: x[:, 1])(round_out)
    round_out_3 = Lambda(lambda x: x[:, 2])(round_out)
    round_out_4 = Lambda(lambda x: x[:, 3])(round_out)
    round_out_5 = Lambda(lambda x: x[:, 4])(round_out)
    transformed_image_1 = Multiply(name=prefix + "_mul1")([round_out_1, transformed_image_1])
    transformed_image_2 = Multiply(name=prefix + "_mul2")([round_out_2, transformed_image_2])
    transformed_image_3 = Multiply(name=prefix + "_mul3")([round_out_3, transformed_image_3])
    transformed_image_4 = Multiply(name=prefix + "_mul4")([round_out_4, transformed_image_4])
    transformed_image_5 = Multiply(name=prefix + "_mul5")([round_out_5, transformed_image_5])

    x = Add(name=prefix + "add")([transformed_image_1, transformed_image_2,
                                  transformed_image_3, transformed_image_4, transformed_image_5])
    return x


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3]//2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])

    return x


def sse_block(prevlayer, prefix):
    conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])

    return conv


def csse_block(x, prefix):
    """
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    """
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


def create_transformer_model(input_shape):
    transformer_model = Sequential()
    #transformer_model.add(Conv2D(3, (1, 1), input_shape=input_shape))
    loc_net = get_loc_net(input_shape=input_shape,
                          transformer_name=args.transformer_type)
    transformer_layer = get_keras_layer(args.transformer_type)
    transformer_model.add(transformer_layer(localization_net=loc_net, output_size=input_shape, input_shape=input_shape))
    return transformer_model


def transformer_block(prev_layer):
    input_shape = K.int_shape(prev_layer)[1:]
    loc_net = get_loc_net(input_shape=input_shape,
                          transformer_name=args.transformer_type)
    transformer_layer = get_keras_layer(args.transformer_type)
    transformed_image = transformer_layer(localization_net=loc_net, output_size=input_shape)(prev_layer)
    return transformed_image

"""
Unet with Mobile net encoder
Uses caffe preprocessing function
"""

def get_unet_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

def get_unet_ddtn_resnet(input_shape):
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # resh_conv = Conv2D()
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    # if args.show_summary:
    #     resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    # transformer_model = create_transformer_model(input_shape)
    # transformer_model = Sequential()
    # transformer_model.add(Conv2D(3, (1, 1), input_shape=input_shape))
    # loc_net = get_loc_net(input_shape=input_shape,
    #                       transformer_name=args.transformer_type)
    # transformer_layer = get_keras_layer(args.transformer_type)
    # transformer_model.add(transformer_layer(localization_net=loc_net, output_size=input_shape))
    # encoded_image = transformer_layer(localization_net=loc_net, output_size=input_shape)(resnet_base.get_layer("input_1").output)
    #encoded_image = transformer_model(resnet_base.get_layer("input_1").output)
    input_image = resnet_base.get_layer("input_1").output
    transformed_image = transformer_block(input_image, input_shape)
    resnet_base.get_layer("conv1")(transformed_image)

    # conv0 = csse_block(input_image, "csse_0")
    # resnet_base.get_layer("conv1")(conv0)
    conv1 = resnet_base.get_layer("activation_1").output
    # conv1 = csse_block(conv1, "csse_1")
    # resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model


def get_csse_unet_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model


# def get_csse_resnet_nt(input_shape):
#     resnet_base = ResNet50(input_shape=input_shape, include_top=False)
#
#     if args.show_summary:
#         resnet_base.summary()
#
#     for l in resnet_base.layers:
#         l.trainable = True
#     conv1 = resnet_base.get_layer("activation_1").output
#     conv1 = csse_block(conv1, "csse_1")
#     nadir_out = classification_branch(conv1, "nadir", 3)
#     tangent_out = classification_branch(conv1, "tangent", 3)
#     conv1 = transformation_branch_3(nadir_out, conv1, "nadir_transform")
#     conv1 = transformation_branch_3(tangent_out, conv1, "tangent_transform")
#     # conv1 = transformer_block(conv1)
#
#     resnet_base.get_layer("max_pooling2d_1")(conv1)
#     conv2 = resnet_base.get_layer("activation_10").output
#     conv2 = csse_block(conv2, "csse_10")
#     resnet_base.get_layer("res3a_branch2a")(conv2)
#     conv3 = resnet_base.get_layer("activation_22").output
#     conv3 = csse_block(conv3, "csse_22")
#     resnet_base.get_layer("res4a_branch2a")(conv3)
#     conv4 = resnet_base.get_layer("activation_40").output
#     conv4 = csse_block(conv4, "csse_40")
#     resnet_base.get_layer("res5a_branch2a")(conv4)
#     conv5 = resnet_base.get_layer("activation_49").output
#     conv5 = csse_block(conv5, "csse_49")
#     resnet_base.get_layer("avg_pool")(conv5)
#
#     up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
#     conv6 = conv_block_simple(up6, 256, "conv6_1")
#     conv6 = conv_block_simple(conv6, 256, "conv6_2")
#     conv6 = csse_block(conv6, "csse_6")
#
#     up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
#     conv7 = conv_block_simple(up7, 192, "conv7_1")
#     conv7 = conv_block_simple(conv7, 192, "conv7_2")
#     conv7 = csse_block(conv7, "csse_7")
#
#     up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
#     conv8 = conv_block_simple(up8, 128, "conv8_1")
#     conv8 = conv_block_simple(conv8, 128, "conv8_2")
#     conv8 = csse_block(conv8, "csse_8")
#
#     up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
#     conv9 = conv_block_simple(up9, 64, "conv9_1")
#     conv9 = conv_block_simple(conv9, 64, "conv9_2")
#     conv9 = csse_block(conv9, "csse_9")
#
#     vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
#     for l in vgg.layers:
#         l.trainable = False
#     vgg_first_conv = vgg.get_layer("block1_conv2").output
#     up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
#     conv10 = conv_block_simple(up10, 32, "conv10_1")
#     conv10 = conv_block_simple(conv10, 32, "conv10_2")
#     conv10 = csse_block(conv10, "csse_o10")
#     conv10 = SpatialDropout2D(0.2)(conv10)
#     x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
#     model = Model(resnet_base.input, [x, nadir_out, tangent_out])
#     return model
def get_csse_resnet_nt(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    nadir_out = classification_branch(conv1, "nadir", 3)
    tangent_out = classification_branch(conv1, "tangent", 3)
    conv1 = transformation_branch(nadir_out, conv1, "nadir_transform")
    conv1 = transformation_branch(tangent_out, conv1, "tangent_transform")
    # conv1 = transformer_block(conv1)

    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, [x, nadir_out, tangent_out])
    return model

def get_csse_resnet_nt_5(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    nadir_out = classification_branch(conv1, "nadir", 5)
    tangent_out = classification_branch(conv1, "tangent", 3)
    conv1 = transformation_branch_5(nadir_out, conv1, "nadir_transform")
    conv1 = transformation_branch_3(tangent_out, conv1, "tangent_transform")
    # conv1 = transformer_block(conv1)

    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, [x, nadir_out, tangent_out])
    return model


def csse_resnet50_fpn(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(resnet_base.input, x)
    return model


def csse_resnet50_fpn_nt(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    nadir_out = classification_branch(conv1, "nadir", 3)
    tangent_out = classification_branch(conv1, "tangent", 3)
    conv1 = transformation_branch_3(nadir_out, conv1, "nadir_transform")
    conv1 = transformation_branch_3(tangent_out, conv1, "tangent_transform")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="prediction")(x)
    model = Model(resnet_base.input, [x, nadir_out, tangent_out])
    return model


def csse_resnet50_fpn_nt_5(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    nadir_out = classification_branch(conv1, "nadir", 5)
    tangent_out = classification_branch(conv1, "tangent", 3)
    conv1 = transformation_branch_5(nadir_out, conv1, "nadir_transform")
    conv1 = transformation_branch_3(tangent_out, conv1, "tangent_transform")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="prediction")(x)
    model = Model(resnet_base.input, [x, nadir_out, tangent_out])
    return model


def resnet50_fpn(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    # conv1 = resnet_base.get_layer("conv1_relu").output
    # conv2 = resnet_base.get_layer("res2c_relu").output
    # conv3 = resnet_base.get_layer("res3d_relu").output
    # conv4 = resnet_base.get_layer("res4f_relu").output
    # conv5 = resnet_base.get_layer("res5c_relu").output
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(resnet_base.input, x)
    return model

def get_csse_hypercolumn_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    hyper = concatenate([conv10,
                         UpSampling2D(size=2)(conv9),
                         UpSampling2D(size=4)(conv8),
                         UpSampling2D(size=8)(conv7),
                         UpSampling2D(size=16)(conv6)], axis=-1)
    hyper = SpatialDropout2D(0.2)(hyper)
    # x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(hyper)
    x = Conv2D(1, (1, 1), name="no_activation_prediction", activation=None)(hyper)
    x = Activation('sigmoid', name="activation_prediction")(x)
    model = Model(resnet_base.input, x)
    return model


def get_simple_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model


def get_ddtn_unet(input_shape):
    img_input = Input(input_shape)
    #if args['transformer_type'] != 'no':  # only construct if we want to use transformers
        # Construct localization network
    # loc_net = get_loc_net(input_shape=input_shape,
    #                       transformer_name=args.transformer_type)
    # loc_net = get_loc_net_func(input_shape=input_shape,
    #                       prevlayer=img_input,
    #                       transformer_name=args.transformer_type)
    # # Add localization network and transformer layer
    loc_net = get_loc_net(input_shape=input_shape,
                          transformer_name=args.transformer_type)
    transformer_layer = get_keras_layer(args.transformer_type)
    # transformer_model.add(transformer_layer(localization_net=loc_net, output_size=input_shape))
    encoded_image = transformer_layer(localization_net=loc_net, output_size=input_shape)(img_input)
    #else:
        #model.add(Lambda(lambda x: x))  # identity layer -> same model structure
    conv1 = conv_block_simple(encoded_image, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model


def get_csse_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    conv1 = csse_block(conv1, "csse_1")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    conv2 = csse_block(conv2, "csse_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    conv3 = csse_block(conv3, "csse_3")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")
    conv5 = csse_block(conv5, "csse_5")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model
"""
Unet with Mobile net encoder
Uses the same preprocessing as in Inception, Xception etc. (imagenet_utils.preprocess_input with mode 'tf' in new Keras version)
"""


def get_unet_mobilenet(input_shape):
    base_model = MobileNet(include_top=False, input_shape=input_shape)

    conv1 = base_model.get_layer('conv_pw_1_relu').output
    conv2 = base_model.get_layer('conv_pw_3_relu').output
    conv3 = base_model.get_layer('conv_pw_5_relu').output
    conv4 = base_model.get_layer('conv_pw_11_relu').output
    conv5 = base_model.get_layer('conv_pw_13_relu').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 192, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 96, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


"""
Unet with Inception Resnet V2 encoder
Uses the same preprocessing as in Inception, Xception etc. (imagenet_utils.preprocess_input with mode 'tf' in new Keras version)
"""


def get_unet_inception_resnet_v2(input_shape):
    base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    conv1 = base_model.get_layer('activation_3').output
    conv2 = base_model.get_layer('activation_5').output
    conv3 = base_model.get_layer('block35_10_ac').output
    conv4 = base_model.get_layer('block17_20_ac').output
    conv5 = base_model.get_layer('conv_7b_ac').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.4)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


def get_vgg_7conv(input_shape):
    img_input = Input(input_shape)
    vgg16_base = VGG16(input_tensor=img_input, include_top=False)
    for l in vgg16_base.layers:
        l.trainable = True
    conv1 = vgg16_base.get_layer("block1_conv2").output
    conv2 = vgg16_base.get_layer("block2_conv2").output
    conv3 = vgg16_base.get_layer("block3_conv3").output
    pool3 = vgg16_base.get_layer("block3_pool").output

    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv1")(pool3)
    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv2")(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv1")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv2")(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv1")(pool5)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv2")(conv6)
    pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(conv6)

    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv1")(pool6)
    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv2")(conv7)

    up8 = concatenate([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv7), conv6], axis=3)
    conv8 = Conv2D(384, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up8)

    up9 = concatenate([Conv2DTranspose(256, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv8), conv5], axis=3)
    conv9 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up9)

    up10 = concatenate([Conv2DTranspose(192, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv9), conv4], axis=3)
    conv10 = Conv2D(192, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up10)

    up11 = concatenate([Conv2DTranspose(128, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv10), conv3], axis=3)
    conv11 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up11)

    up12 = concatenate([Conv2DTranspose(64, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv11), conv2], axis=3)
    conv12 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up12)

    up13 = concatenate([Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv12), conv1], axis=3)
    conv13 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up13)

    conv13 = Conv2D(1, (1, 1))(conv13)
    conv13 = Activation("sigmoid")(conv13)
    model = Model(img_input, conv13)
    return model


def make_model(input_shape):
    network = args.network
    if network == 'resnet50':
        return get_unet_resnet(input_shape)
    if network == 'ddtn_resnet50':
        return get_unet_ddtn_resnet(input_shape)
    elif network == 'csse_resnet50':
        return get_csse_unet_resnet(input_shape)
    elif network == 'csse_resnet50_nt':
        return get_csse_resnet_nt(input_shape)
    elif network == 'hypercolumn_resnet':
        return get_csse_hypercolumn_resnet(input_shape)
    elif network == 'inception_resnet_v2':
        return get_unet_inception_resnet_v2(input_shape)
    elif network == 'mobilenet':
        return get_unet_mobilenet(input_shape)
    elif network == 'vgg':
        return get_vgg_7conv(input_shape)
    elif network == 'simple_unet':
        return get_simple_unet(input_shape)
    elif network == 'ddtn_unet':
        return get_ddtn_unet(input_shape)
    elif network == 'csse_unet':
        return get_csse_unet(input_shape)
    elif network == 'csse_unet':
        return get_csse_unet(input_shape)
    elif network == 'resnet50_fpn':
        return resnet50_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'csse_resnet50_fpn':
        return csse_resnet50_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'csse_resnet50_fpn_nt':
        return csse_resnet50_fpn_nt(input_shape, channels=1, activation="sigmoid")
    elif network == 'csse_resnet50_fpn_nt_5':
        return csse_resnet50_fpn_nt_5(input_shape, channels=1, activation="sigmoid")
    else:
        raise ValueError("Unknown network")
