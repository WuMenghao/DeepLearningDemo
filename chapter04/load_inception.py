# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import scipy
import tensorflow as tf

# with tf.Graph() as graph:
# with tf.InteractiveSession(graph=graph) as sess:

# 由于在训练Inception 模型的时候，已经做了减去均值的预处理，因此应该使用
# 同样的预处理方法，才能保持输入的一致。 此处使用的Inception 模型减去
# 的是一个固定的均值117，所以在程序中也定义7imagenet_ mean= 117，并
# 用t_input 减去imagenet_mean。
IMAGE_NET_MEAN = 117.0

def save_image(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('Image saved: %s' % img_name)

if __name__ == '__main__':

    # (1)导入神经网络
    model_fn = 'tensorflow_inception_graph.pb'
    with tf.gfile.FastGFile('trained/tensorflow_inception_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # (2)定义input
    t_input = tf.placeholder(np.float32, name='input')
    # (3)图像数据处理
    # expand_dims是加维度，从［height, width, channel］变成[1, height, width, channel]
    # Inception模型需要的输入格式却是[batch,height, width, channel]
    t_preprocessed = tf.expand_dims(t_input - IMAGE_NET_MEAN, 0)
    # (4)将图像数据输入模型
    tf.import_graph_def(graph_def, {'input': t_preprocessed})
    # (5)找到所有的卷积层
    with tf.Graph().as_default() as graph:
        layers = [op.name for op in graph.get_operations()
                  if op.type == 'Conv2D' and 'import/' in op.name]
        print('Number of layers', len(layers))
        print('Shape of mixed4d_3x3_bottleneck_pre_relu : %s' %
              (str(graph.get_tensor_by_name(
                  'import/mixed4d_3x3_bottleneck_pre_relu:0').get_shape())))
