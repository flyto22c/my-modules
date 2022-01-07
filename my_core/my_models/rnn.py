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

class MyRnn(Model):
    def __init__(self,
                 units,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 ):
        ''' 1.Define your layers in __init__() .
            2.Do not use tf.constant().
        '''
        super(MyRnn, self).__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.rnn_cell = layers.SimpleRNNCell(self.units, dropout=dropout, recurrent_dropout=recurrent_dropout)

    def call(self, inputs, training=None, mask=None):
        # inputs: shape=(b,seq_len,emb_dim)\
        # h_t:shape=(b,h_dim)\
        print(inputs.shape)
        h_t = [tf.zeros(shape=(inputs.shape[0], self.units), dtype=tf.float32)]
        h_t_list = []
        for word in tf.unstack(inputs, axis=1):
            # word:shape=(b, emb_dim)\
            # o_t, [h_t] = rnn_cell(x_t, [h_t-1])
            _, h_t = self.rnn_cell(word, h_t, training=training)
            h_t_list.append(h_t[0])
        if self.return_sequences:
            out = tf.stack(h_t_list, axis=1)
        else:
            out = h_t_list[-1]
        return out
