import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, datasets
from tensorflow.keras import layers, optimizers, losses, metrics
from tensorflow.keras import models, regularizers, initializers

'''
    1. Use Layer as much as possible makes Model more clarified .
'''
class MyDense(layers.Layer):
    def __init__(self, units=32):
        '''
        Description:
            1.Define [trainable=False] Weight or Args , which don't depend on input_shape.
                e.g.: self.total = tf.Variable(initial_value=tf.zeros((units,)),trainable=False)
            2.Do not use tf.constant()
        Args:
            units: N_new
        '''
        super(MyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        '''
        Description:
            1.Define [trainable=True] weight and Args , which depend on input_shape.
            2.How to define tensor influence tf.clip_by_norm() .
        Args:
            input_shape:  (b, N)
        '''
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units)),
                             dtype=tf.float32,
                             trainable=True)
        self.w1 = tf.Variable(tf.random.truncated_normal(shape=(input_shape[-1], self.units), stddev=0.1),
                              dtype=tf.float32,
                              trainable=True)
        self.w2 = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer=initializers.TruncatedNormal(stddev=0.1),
                                  dtype=tf.float32,
                                  trainable=True)
        ''' self.add_variable() ＝ self.add_weight()
            initializers.truncated_normal\Lecun\Xavier\he
        '''

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,)),
                             dtype=tf.float32,
                             trainable=True)
        self.b1 = tf.Variable(tf.zeros(shape=(self.units)),
                              dtype=tf.float32,
                              trainable=True)
        self.b2 = self.add_weight(shape=(self.units,),
                                  initializer=initializers.zeros,
                                  dtype=tf.float32,
                                  trainable=True)

    def call(self, inputs, training=None, mask=None):
        '''
        Description:
            1.Compute porcess in call() .
            2.Update [trainable=False] Weights or Args  manually during call().
                Examples: self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        Args:
            inputs: tensor(b,N), A tensor or list of tensors.
            training:
                True: [train mode] data has dim B.
                False: [infer mode] data has dim B.
                Description: attention if exists [trainable=False] args .
            mask: boolean tensor encoding masked timesteps in the input, used in RNN layers .
                A mask or list of masks or None (no mask), often used in NLP tasks.
                eg. https://www.cnblogs.com/databingo/p/9339175.html
        '''
        return inputs @ self.w + self.b

    def get_config(self):
        ''' 1.This method is used when saving the layer or a model that contains this layer.
        '''
        pass
