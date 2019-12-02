# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/12/2

from __future__ import print_function

import numpy as np
import scipy
import tensorflow as tf
import scipy.misc

'''
最终的DeepDream模型还
需要对图片添加一个背景。具体应该怎么做呢？真实，之前是从image_noise
开始优化图像的，现在使用一张背景图像作为起点对图像进行优化就可以了。
'''

IMAGE_NET_MEAN = 117.0
USED_LAYER = 'mixed4c'
USED_LAYER_TENSOR_NAME = 'import/%s:0' % USED_LAYER
CHANNEL = 139  # 可以尝试不同的通道，如channel=99时


def save_image(img_array, img_name):
    scipy.misc.imsave(img_name, img_array)
    print('Image saved: %s' % img_name)


def resize(image, shape):
    """
    按比例缩放图片：在实际工程中为了加快图像的收敛速度，采用先生成小尺寸，再将图片放大的方法
    :param image: original image
    :param shape: image shape
    :return: resized image
    """
    min = image.min()
    max = image.max()
    image = (image - min) / (max - min) * 255
    image = np.float32(scipy.misc.imresize(image, shape))
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


def render_deepdream(sess, input, tensor, image,
                     iter_n=10, step=1.5, octaves=4, octave_scale=1.4):
    # (1)定义目标和梯度
    t_score = tf.reduce_mean(tensor)
    t_grad = tf.gradients(t_score, input)[0]

    # (2)同样将图像进行金字塔分解 此时提取高频、低频的方法比较简单。直接缩放就可以
    img_temp = image
    octave_list = []
    for i in range(octaves - 1):
        shape = img_temp.shape[:2]
        low_shape = np.int32(np.float32(shape) / octave_scale)
        low = resize(img_temp, low_shape)
        high = img_temp - resize(low, shape)
        img_temp = low
        octave_list.append(high)

    # (3)先生成低频的图像，再依次放大并加上高频
    for octave in range(octaves):
        if octave > 0:
            high = octave_list[-octave]
            img_temp = resize(img_temp, high.shape[:2]) + high
        for i in range(iter_n):
            g = calc_grad_tiled(sess, img_temp, input, t_grad)
            img_temp += g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')

        img_temp = img_temp.clip(0, 255)
    save_image(img_temp, 'result/deepdream.jpg')


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
            t_input = load_inception()
            # (1)获取layer_output
            layer_output = tf.square(graph.get_tensor_by_name(USED_LAYER_TENSOR_NAME))
            # (2)读取背景图片
            image = np.float32(scipy.misc.imread('test.jpg'))
            # (3)调用render_native函数渲染
            render_deepdream(sess=sess, input=t_input, tensor=layer_output,
                                 image=image)


if __name__ == '__main__':
    tf.app.run()
