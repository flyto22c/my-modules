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
''' 
    1. Use API but Model, because plot_model() does not work in Model.
    2. `multiple` problem with TF version:https://github.com/keras-team/keras/issues/13782   
    3. `multiple` problem with Model.build() :https://github.com/tensorflow/tensorflow/issues/29132
'''
class MyModel(Model):
    def __init__(self):
        '''
        Description:
            1.Define your layers in __init__() .
            2.Do not use tf.constant().
        '''
        super(MyModel, self).__init__()
        # Define layers need mode [training=False\True].
        self.dense1 = layers.Dense(4, activation=tf.nn.relu)
        # Define layers doesn't need mode [training=False\True].
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        '''
        Description: Implement the model's forward pass in call().
        Args:
            inputs: A tensor or list of tensors.
            training:
                True: [train mode] data has dim B.(Dropout\BatchNormal\RNN)
                False: [infer mode] data has dim B.
                Description: attention if exists [trainable=False] args .
            mask:
                Boolean tensor encoding masked timesteps in the input, used in RNN layers .
                A mask or list of masks or None (no mask), often used in NLP tasks.
                eg. https://www.cnblogs.com/databingo/p/9339175.html
        '''
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x
    def build_graph(self, input_shape):
        '''
        Description:
            1.Solve the problem of `multiple` in build() .
            2.Attention: Input may be mutiple tensors ,input_shape[None,n]
        '''
        # build() before summary() .
        self.build(input_shape=input_shape)

        # call() make `multiple` to shape .
        inputs = tf.keras.Input(batch_input_shape=input_shape)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)

