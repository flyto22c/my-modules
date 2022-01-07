
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        # filter_num:通道数量。
        # stride：步长。
        super(BasicBlock, self).__init__()
        # 基本单元：conv->BN->POOL->RELU
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # short-cut是一个(1,1)的卷积层。
        if stride != 1:  # 步长不为 1 时
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:            # 步长为 1 时
            self.downsample = lambda x:x


    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)
        identity = self.downsample(inputs)
        # 这个是什么用法？add
        output = layers.add([out, identity])

        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100): # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 基本单元：conv->BN->POOL->RELU
        # self.in_t = layers.Input(shape=(32,32,3))
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
            ])

        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        # self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        # self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output: [b, 512, h, w],
        # 全局池化层。
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)



    def call(self, inputs, training=None , mask=None):
        # (None,32,32,3)
        x = self.stem(inputs,training=training)
        x = self.layer1(x,training=training)
        x = self.layer2(x,training=training)
        # x = self.layer3(x,training=training)
        # x = self.layer4(x,training=training)

        # [b, c] todo:这里如果有training怎么样？报错？
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)
        return x
    # 类的方法。
    # def build_resblock(self, filter_num, blocks, stride=1):
    #
    #     res_blocks = Sequential()
    #     # may down sample

    #     # 这里有下采样功能。
    #     res_blocks.add(BasicBlock(filter_num, stride))
    #     # 后续的 block 不需要下采样功能。
    #     for _ in range(1, blocks):
    #         res_blocks.add(BasicBlock(filter_num, stride=1))
    #     return res_blocks

def resnet18():
    # Adjust the unit_num by your needs .
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
