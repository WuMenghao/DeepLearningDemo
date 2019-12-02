# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/11/29

from __future__ import print_function

import numpy as np
import scipy
import tensorflow as tf
import scipy.misc
from functools import partial

'''
第4.2.3 节中生成图像的高频成分太多，而希望图像
的低频成分应该多一些，这样生成的图像才会更加“柔和”。
如何让图像具高更多的低频成分而不是高频成分？一种方法是针对高
频成分加入损失，这样图像在生成的时候就会因为新加入损失的作用而发生
改变。但加入损失会导致计算量和收敛步数的增大。 此处采用另一种方法·
放大低频的梯度。之前生成图像时，使用的梯度是统一的。如果可以对梯度
作分解，将之分为“高频梯度”“低频梯度’＼再人为地去放大“低频梯度”，
就可以得到较为柔和的固像了。
在具体实践上，使用拉普拉斯金字塔（Laplacian Pyramid ）对图像进行
分解。这种算法可以把圄片分解为多层，如图4-5所示。底层的levell、 level2
就对应图像的高频成分，而上层的level3、 level4 对应图像的低频成分。可
以对梯度也做这样的分解。分解之后，对高频的梯度和低频的梯度都做标准
化，可以让梯度的低频成分和高频成分差不多，表现在图像上就会增加圄像
的低频成分，从而提高生成图像的质量。通常称这种方法为拉普拉斯金字塔
梯度标准化（Laplacian Pyramid Gradient Normalization ）。
'''
IMAGE_NET_MEAN = 117.0
USED_LAYER = 'mixed4d_3x3_bottleneck_pre_relu'
USED_LAYER_TENSOR_NAME = 'import/%s:0' % USED_LAYER
CHANNEL = 139 # 可以尝试不同的通道，如channel=99时

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)


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


def laplace_split(image):
    """
    拉普拉斯法区分高频成分和低频成分
    """
    with tf.name_scope('laplace_split'):
        # 做过一次卷积相当于一次"平滑"得到低频成分
        low = tf.nn.conv2d(image, k5x5, [1, 2, 2, 1], 'SAME')
        # 低频成分缩放到原始大小得到low_ori，再用原挺圈像img减去low_ori，就得到高频成分high
        low_ori = tf.nn.conv2d_transpose(low, k5x5 * 4, tf.shape(image), [1, 2, 2, 1])
        high = image - low_ori
        return low, high


def laplace_split_n(image, n):
    """
    将image分为n 层拉普拉斯金字塔
    """
    levels = []
    for i in range(n):
        # 高频部分保存到levels 低频部分继续分解
        image, high = laplace_split(image)
        levels.append(high)
    levels.append(image)
    return levels[::-1]


def laplace_merge(levels):
    """
    将拉普拉斯金字塔还原到原始图像
    """
    image = levels[0]
    for high in levels[1:]:
        with tf.name_scope('merge'):
            image = tf.nn.conv2d_transpose(
                image, k5x5 * 4, tf.shape(high), [1, 2, 2, 1]) + high
    return image


def normalize_std(image, eps=1e-10):
    """
    对image 做标准化
    """
    with tf.name_scope('normallize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(image)))
        return image / tf.maximum(std, eps)


def laplace_normalize(image, scale=4):
    """
    拉普拉斯金字塔标准化
    :param image:
    :param scale:
    :return:
    """
    # (1)将图片扩维
    image = tf.expand_dims(image, 0)
    # (2)拉普拉斯法区分高频成分和低频成分
    levels = laplace_split_n(image, scale)
    # (3)每一层都做一次normalize_std
    levels = list(map(normalize_std, levels))
    # (4)将拉普拉斯金字塔还原到原始图像
    out = laplace_merge(levels)
    return out[0, :, :, :]


def transfer_func(*argtypes):
    """
    包装成
    :param argtypes:
    :return:
    """
    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), kw.get('session'))

        return wrapper

    return wrap


def render_laplace_normal(sess, input, tensor, image,
                          iter_n=10, step=1.0, octaves=3, octave_scale=1.4, lap_scale=4):
    # (1)定义目标和梯度
    t_score = tf.reduce_mean(tensor)
    t_grad = tf.gradients(t_score, input)[0]
    # (2)将lap_normalize转换为正常函数
    lap_norm_func = transfer_func(np.float32)(partial(laplace_normalize, scale=lap_scale))

    img_temp = image.copy()
    for octave in range(octaves):
        if octave > 0:
            img_temp = resize_ratio(img_temp, octave_scale)
        for i in range(iter_n):
            g = calc_grad_tiled(sess, img_temp, input, t_grad)
            g = lap_norm_func(g)
            img_temp += g * step
            print('.', end='')
    save_image(img_temp, 'result/lapnorm_my_%s_%s.jpg' % (USED_LAYER, CHANNEL))


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
            layer_output = graph.get_tensor_by_name(USED_LAYER_TENSOR_NAME)
            # (2)定义原始的图像噪声
            img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
            # (3)调用render_native函数渲染
            render_laplace_normal(sess=sess, input=t_input, tensor=layer_output[:, :, :, CHANNEL],
                                  image=img_noise, iter_n=20)


if __name__ == '__main__':
    tf.app.run()
