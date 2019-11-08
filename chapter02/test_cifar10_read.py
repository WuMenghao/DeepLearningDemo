# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:42:24 2019

@author: Administrator
"""

import tensorflow as tf
import os
import cifar10_input
import scipy

def inputs_origin(data_dir):
    # 读入训练图
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1,6)]
    # 判断文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: '+f)
    # 读取文件名queue
    filename_queue = tf.train.string_input_producer(filenames)
    # 使用cifar10_input读取uint8image图像
    read_input = cifar10_input.read_cifar10(filename_queue)
    # 对图像进行reshape
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    # 返回的reshape_image是一张图片的tensor
    return reshaped_image

if __name__ == '__main__':
    # 读取图片队列
    reshape_image=inputs_origin('cifar10_data/cifar-10-batches-bin')
    with tf.Session() as sess:
        # start_queue_runners
        threads = tf.train.start_queue_runners(sess=sess)
        # init variables
        sess.run(tf.global_variables_initializer())
        # mkdir
        if not os.path.exists('cifar10_data/raw/'):
            os.makedirs('cifar10_data/raw/')
        #保存30张图片
        for i in range(30):
            # one image
            image_array = sess.run(reshape_image)
            # save image
            scipy.misc.imsave('cifar10_data/raw/%d.jpg' % i,image_array)