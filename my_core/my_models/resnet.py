import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, datasets
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics, models, regularizers, initializers
from tensorflow.keras.utils import plot_model
assert tf.__version__.startswith('2.')


class ResnetUnit(Model):
    def __init__(self, filter_num, stride_shape=(1, 1)):
        ''' 1.Define your layers in __init__() .
            2.Do not use tf.constant()。
        Args:
            filters: filter_num
            strides: stride_shape
        '''
        super(ResnetUnit, self).__init__()
        # Uint:conv->bn->relu
        self.conv1 = layers.Conv2D(filter_num, (3, 3), padding='same', strides=stride_shape)
        self.bn1 = layers.BatchNormalization(axis=-1, center=True, scale=True)
        # Uint:conv->bn->relu
        self.conv2 = layers.Conv2D(filter_num, (3, 3), padding='same', strides=(1, 1))
        self.bn2 = layers.BatchNormalization(axis=-1, center=True, scale=True)
        if stride_shape != (1, 1):
            self.downsample = layers.Conv2D(filter_num, (1, 1), padding='same', strides=stride_shape)
        else:
            # Layers is a func .
            self.downsample = lambda x: x

    def call(self, inputs, training=None, mask=None):
        ''' Implement the model's forward pass in call().
            Tips: handle training\mask in the last step.
        Args:
            inputs: shape=(b,h,w,c)
            training: boolean, whether the call is in [infer mode] or [train mode].
            mask: boolean tensor encoding masked timesteps in the input, used in RNN layers .
                A mask or list of masks or None (no mask).
        '''
        # Uint:conv->bn->relu
        h = self.conv1(inputs)
        h = self.bn1(h, training=training)
        h = tf.nn.relu(h)
        # Uint:conv->bn->relu
        h = self.conv2(h)
        h = self.bn2(h, training=training)
        identity = self.downsample(inputs)
        h = h + identity
        h = tf.nn.relu(h)

        return h


class Resnet(Model):
    def __init__(self, unit_num_list, class_num=100):
        ''' 1.Define your layers in __init__() .
            2.Do not use tf.constant()。
        '''
        super(Resnet, self).__init__()
        self.stem_conv = layers.Conv2D(64, (3, 3), strides=(1, 1))
        self.stem_bn = layers.BatchNormalization(axis=-1, center=True, scale=True)
        self.stem_max_pool = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')

        self.resnet_block1 = self.build_resnet_block(unit_num_list[0], filter_num=64, stride_shape=(1, 1))
        self.resnet_block2 = self.build_resnet_block(unit_num_list[1], filter_num=128, stride_shape=(2, 2))
        self.resnet_block3 = self.build_resnet_block(unit_num_list[2], filter_num=256, stride_shape=(2, 2))
        self.resnet_block4 = self.build_resnet_block(unit_num_list[3], filter_num=512, stride_shape=(2, 2))

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(class_num)

    def call(self, inputs, training=None, mask=None):
        ''' Implement the model's forward pass in call().
        Args:
            inputs: A tensor or list of tensors.
            training: boolean, whether the call is in [infer mode] or [train mode].
            mask: boolean tensor encoding masked timesteps in the input, used in RNN layers .
                A mask or list of masks or None (no mask).
        '''
        # stem:conv->bn->relu -> pool
        x = self.stem_conv(inputs, training=training)
        x = self.stem_bn(x, training=training)
        x = tf.nn.relu(x, training=training)
        x = self.stem_max_pool(x, training=training)
        # Resnet
        x = self.resnet_block1(x, training=training)
        x = self.resnet_block2(x, training=training)
        x = self.resnet_block3(x, training=training)
        x = self.resnet_block4(x, training=training)
        # MLP
        x = self.avgpool(x, training=training)
        x = self.fc(x, training=training)
        return x

    def build_resnet_block(self, unit_num, filter_num, stride_shape):
        resnet_block = Sequential()
        for i in unit_num:
            if i == 0:
                resnet_block.add(ResnetUnit(filter_num=filter_num, stride_shape=stride_shape))
            else:
                resnet_block.add(ResnetUnit(filter_num=filter_num, stride_shape=(1, 1)))
        return resnet_block
