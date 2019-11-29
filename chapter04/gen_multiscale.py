# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.misc

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
    scipy.misc.imsave(img_name, img_array)
    print('Image saved: %s' % img_name)


def resize_ratio(image, ratio):
    """
    按比例缩放图片：在实际工程中为了加快图像的收敛速度，采用先生成小尺寸，再将图片放大的方法
    :param image: original image
    :param ratio: scaling ratio
    :return: resized image
    """
    height, width = image.shape[:2]
    min = image.min()
    max = image.max()
    image = (image - min) / (max - min) * 255
    image = np.float32(scipy.misc.imresize(image, ratio))
    image = image / 255 * (max - min) + min
    return image


def calc_grad_tiled(sess, image, t_input, t_grad, tile_size=512):
    """
    对任意大小的图片计算梯度并应用
    :param sess: tensorflow session
    :param image: resize image
    :param t_input: placeholder of input
    :param t_grad: gradient of inception
    :param tile_size: size of each tile
    :return: result image
    """
    # (1)图形变换在x,y 轴上整体运动,防止tile 产生边缘效应
    sz = tile_size
    height, width = image.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    image_shift = np.roll(a=image, shift=sx, axis=1)
    image_shift = np.roll(a=image_shift, shift=sy, axis=0)
    # (2)每次只对tile_size * tile_size 大小的图像计算梯度, 避免内存问题
    grad = np.zeros_like(image)
    for y in range(0, max(height - sz // 2, sz), sz):
        for x in range(0, max(height - sz // 2, sz), sz):
            sub_img = image_shift[y: y + sz, x: x + sz]
            sub_grad = sess.run(t_grad, {t_input: sub_img})
            grad[x: x + sz, y: y + sz] = sub_grad
    # (3)将图形在x,y 轴上移回
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_multi_scale(sess, t_input, tensor, img0,
                       iter_n=10, step=1, octave_n=3, octave_scale=1.4):
    """
    生成大尺寸DeepDream图片
    """
    t_score = tf.reduce_mean(tensor)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            img = resize_ratio(image=img, ratio=octave_scale)
        for i in range(iter_n):
            g = calc_grad_tiled(sess=sess, image=img, t_input=t_input, t_grad=t_grad)
            g /= g.std() + (1e-8)
            img += g * step
            print(".", end=" ")
    save_image(img, 'result/multiscale_my.jpg')


def load_inception():
    """
    导入神经网络
    """
    # (1)导入神经网络
    with tf.gfile.FastGFile('model/tensorflow_inception_graph.pb', 'rb') as f:
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
    return t_input


def main(_):
    with tf.Graph().as_default() as graph:
        with tf.InteractiveSession(graph=graph).as_default() as sess:
            # 导入神经网络模型
            t_input= load_inception()
            # (1)获取layer_output
            layer_output = graph.get_tensor_by_name(USED_LAYER_TENSOR_NAME)
            # (2)定义原始的图像噪声
            img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
            # (3)调用render_native函数渲染
            render_multi_scale(sess=sess, t_input=t_input, tensor=layer_output[:, :, :, CHANNEL],
                               img0=img_noise, iter_n=20)


if __name__ == '__main__':
    tf.app.run()
