# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:18:19 2019
手写数字识别:
    Softmax模型 + 交叉熵
@author: Administrator
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
预测
'''
def prediction(sess,y,y_):
    correct_predoction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predoction,tf.float32))
    print("test accuracy %g" % sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

if __name__ == '__main__':
    # 从MNIST_data/中读取MNIST数据。这条语句在数据不存在时，会自动执行下载
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    #看前二十张图片的lable
    #for i in range(20):
    #    one_lable = mnist.train.labels[i,:]
    #    label=np.argmax(one_lable)
    #    print('mnist_train_%d.jpg lable: %d' % (i,label))
    '''(1)定义模型和输入'''
    '''x :占位符，表示识别的图片'''
    x = tf.placeholder(tf.float32,[None, 784])
    '''W :Softmax模型的参数,将一个784维的输入转换为一个10维的输出'''
    W = tf.Variable(tf.zeros([784,10]))
    '''b :Softmax模型的参数,一般叫偏执项'''
    b = tf.Variable(tf.zeros([10]))
    '''y :表示模型输出'''
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    
    '''y_ :占位符,实际的图像标签'''
    y_ = tf.placeholder(tf.float32,[None,10])
    
    '''
    (2)计算交叉熵损失
    至此，得到两个tensorflow重要参数 y和y_
        y : 模型的输出
        y_: 实际图像标签
        (注:y_是one-hot code表示的，
        独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制)
    
    下面根据y和y_构造 交叉熵 损失
    '''
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_)
    
    '''
    (3)使用梯度下降法正对模型参数(W和b)进行优化
    '''
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    '''
    (4)训练
        每次取mnist.train中的100个训练数据，训练1000次
        batch_xs [100,784]
        batch_ys [100,10]
    '''
    with tf.Session() as sess:
        #初始化变量
        tf.global_variables_initializer().run()
        for _ in range(1000):
            #训练
            batch_xs,batch_ys = mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
            #预测
            prediction(sess,y,y_)
    
    #保存
    #tf.train.Saver().save(sess,'./save/model01')


