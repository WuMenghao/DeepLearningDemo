# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import scipy.misc
import tensorflow as tf

# with tf.Graph() as graph:
# with tf.InteractiveSession(graph=graph) as sess:

# 由于在训练Inception 模型的时候，已经做了减去均值的预处理，因此应该使用
# 同样的预处理方法，才能保持输入的一致。 此处使用的Inception 模型减去
# 的是一个固定的均值117，所以在程序中也定义7imagenet_ mean= 117，并
# 用t_input 减去imagenet_mean。
IMAGE_NET_MEAN = 117.0
USED_LAYER = 'mixed4d_3x3_bottleneck_pre_relu'
USED_LAYER_TENSOR_NAME = 'import/%s:0' % USED_LAYER
CHANNEL = 139


def save_image(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def render_naive(sess, t_input, t_obj, img0, iter_n=20, step=1.0):
    """
    生成原始DeepDream图片处理函数
    """
    # (1)计算优化目标
    t_score = tf.reduce_mean(t_obj)
    # (2)计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]
    # (3)生产新的图像
    img = img0.copy()
    for i in range(iter_n):
        # 计算score和gradients 对 image 应用梯度 ,step可看作学习率
        g, score = sess.run([t_grad, t_score], {t_input: img})
        g /= g.std() + 1e-8
        img += g * step
        print('score(mean）＝%f' % (score))
    save_image(img, 'result/naive_my.jpg')


def load_inception(graph):
    """
    导入神经网络
    :param graph:
    :return:
    """
    # (1)导入神经网络
    model_file = 'model/tensorflow_inception_graph.pb'
    with tf.gfile.FastGFile(model_file, 'rb') as f:
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
    # layers = [op.name for op in graph.get_operations()
    #           if op.type == 'Conv2D' and 'import/' in op.name]
    # print('Number of layers', len(layers))
    # print('Shape of %s : %s' % (USED_LAYER, str(graph.get_tensor_by_name(
    #     USED_LAYER_TENSOR_NAME).get_shape())))
    return t_input


def main(_):
    with tf.Graph().as_default() as graph:
        with tf.InteractiveSession(graph=graph).as_default() as sess:
            # (1)导入神经网络模型
            t_input= load_inception(graph)
            # (2)获取layer_output
            layer_output = graph.get_tensor_by_name(USED_LAYER_TENSOR_NAME)
            # (3)定义原始的图像噪声
            img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
            # (4)调用render_native函数渲染
            render_naive(sess, t_input, layer_output[:, :, :, CHANNEL], img_noise, iter_n=20)



if __name__ == '__main__':
    tf.app.run()
