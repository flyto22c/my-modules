#!/usr/bin/env python
# -*-coding:utf-8-*-
'''
Author: pengfeizhang
DateTime: 2021/7/02/002
Description: Descripe why but what.
'''
''' Step_01: import __file__package
'''
''' Step_02: import inner_package
    import xx
    import xx  as xx
    from   xx  import xx
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, datasets
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics, models, regularizers, initializers
from tensorflow.keras.utils import plot_model

''' Step_03: import my_package
    import my_xx
    import my_xx as xx
    from   my_xx import xx
'''
from my_core import my_models, my_layers

''' Step_04: Head handle.
'''
assert tf.__version__.startswith('2.')

# ---------------------------------------------------------------------------------

def load_model():
    """ load_model"""
    ''' Step_01: Define layers.
    '''
    conv_net = Sequential([
        # 5 units of conv + max pooling
        # unit 1
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 2
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 3
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 4
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 5
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
    ], name='z_covnet')
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),
    ], name='z_fcnet')
    ''' Step_02: Forward process.
    '''
    in_tensor = layers.Input(shape=(32, 32, 3), name='zpf_in')

    x = conv_net(in_tensor)  # forward process can not have args name='' .
    x = tf.reshape(x, [-1, 512], name='zpf_flat')  # layer define can have args name='' .
    x = tf.nn.relu(x, name='zpf_relu')
    out_tensor = fc_net(x)
    ''' Step_03: Build model.
    '''
    model = tf.keras.Model(inputs=[in_tensor],
                           outputs=[out_tensor])
    ''' Step_04: Observe model structure.
        Pre trained model can summary() and plot_model() directly.
        inception_v3 = keras.applications.inception_v3.InceptionV3(include_top=False,classes=500)
        inception_v3.summary()
        plot_model(inception_v3, to_file='inception_v3.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    '''
    model.summary()
    plot_model(model, to_file='model-3.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    return model


def data_generator():
    (x, y), _ = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    train_generator = tf.data.Dataset.from_tensor_slices((x, y))
    train_generator = train_generator.batch(128)
    return train_generator

def main():
    ''' 使用main()函数避免全局变量的污染。 '''
    PARAMS = {
        'EPOCH_NUM': 30,
        'LR': 1e-3,
    }
    # 如果有多个网络部分：variables = conv_net.trainable_variables + fc_net.trainable_variables
    model = load_model()

    optimizer = optimizers.SGD(learning_rate=0.001)

    for epoch in range(30):
        # 最好每个 EPOCH 都打乱一下。
        t0 = time.time()
        ''' Compute train time
            -不能放在epoch外层循环，否则不是计算固定个(100)batch的时间。
            -最后一个batch_size不足不drop掉，会减少计算固定个(100)batch的时间。
            -最后batch数量不达到固定值(100)，t0会在下一个EPOCH重置。
        '''
        list_temp_loss = []
        ''' Record history loss.
            -可以放在epoch外层循环，因为会计算list长度。
            -最后一个batch_size不足不drop掉，不会影响平均loss\metrics计算，因为是除以list的长度。
            -最后batch数量不达到固定值(100)，不会影响整体曲线图，因为平均loss\metrics是除以list的长度
        '''
        for step, (batch_x, batch_y) in enumerate(data_generator):

            with tf.GradientTape() as tape:
                # 1: 计算图过程(网络连接)。
                batch_y_pred = model(batch_x, training=True)  # Attention: train mode .
                # batch_y_pred = h2 @ w3 + b3
                # 2: loss计算.
                loss = tf.reduce_sum(tf.square(batch_y_pred - batch_y)) / batch_x.shape[0]
                # 使用正则化loss:  一般只对w，不对b做正则化。除以2可以方便计算，最好除以M取均值。
                loss_regularization = []
                for p in model.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
                loss = loss + 0.0001 * loss_regularization

            list_temp_loss.append(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            # warning: if grads is None, maybe args is Tensor but Variable.
            # grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
            # grads_norm = [ tf.norm(g) for g in grads ]
            grads = [tf.clip_by_norm(g, 15) for g in grads]
            # grads, _ = tf.clip_by_global_norm(grads, 15)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # w1.assign_sub(lr * grads[0])
            # b1.assign_sub(lr * grads[1])

            ''' Record log info. '''
            batch_idx = step + 1  # batch_idx从1开始，避免step从0开始第一次计算只有一个batch。
            if batch_idx % 100 == 0:
                t1 = time.time()
                delta_time = t1 - t0  # seconds.
                t0 = time.time()
                print('{0:.4f}s / 100 batch'.format(delta_time))

                ''' 重新计算loss，不要有向前传播的影响，例如影响到RNN中的隐藏状态。
                    train_loss: 可以使用平均，避免忽大忽小；
                    train_metrics: 使用平均值会增加一点计算量；因为没有传播过程，有metrics计算过程。
                    val_loss\ val_metrics: 使用平均值会增加很大计算量；有传播过程，有metrics计算过程。
                '''
                train_loss = sum(list_temp_loss) / len(list_temp_loss)
                out = model(batch_x, training=False)  # Attention: infer mode .
                print(epoch, step, 'loss:', loss.numpy())

if __name__ == '__main__':
    main()